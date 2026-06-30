# ingest_data.py
"""
PDF indexing pipeline (PyMuPDF4LLM + OCR)
- 1) PyMuPDF4LLM: convert PDF text to Markdown.
- 2) OCR embedded images to enrich extracted text.
- 3) OCR rendered pages as a fallback for scanned or text-poor PDFs.
- Normalize results into LangChain Documents, split into chunks, embed with OpenAI, and save to FAISS.
"""

from __future__ import annotations

import os
import io
import shutil
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fitz
import pymupdf4llm
from PIL import Image
import pytesseract

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

# ========= Tunable parameters =========
OCR_MIN_W = getattr(config, "OCR_MIN_W", 200)          # Minimum image width for OCR.
OCR_MIN_H = getattr(config, "OCR_MIN_H", 200)          # Minimum image height for OCR.
PAGE_OCR_ENABLE = getattr(config, "PAGE_OCR_ENABLE", True)   # Enable rendered-page OCR fallback.
PAGE_OCR_SCALE = getattr(config, "PAGE_OCR_SCALE", 2.0)      # Render scale. 2.0 is roughly 200-300dpi.


def _configure_tesseract() -> str:
    requested_lang = getattr(config, "OCR_LANG", os.getenv("TESSERACT_LANG", "eng"))
    explicit_cmd = getattr(config, "TESSERACT_CMD", os.getenv("TESSERACT_CMD", ""))
    explicit_data = getattr(config, "TESSDATA_PREFIX", os.getenv("TESSDATA_PREFIX", ""))

    candidate_cmds = [
        explicit_cmd,
        shutil.which("tesseract") or "",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
    ]

    tesseract_cmd = next((path for path in candidate_cmds if path and Path(path).exists()), "")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    else:
        print("[warn] Tesseract executable was not found. Set TESSERACT_CMD.")

    candidate_data_dirs = [
        explicit_data,
        str(Path(tesseract_cmd).parent / "tessdata") if tesseract_cmd else "",
        r"C:\Program Files\Tesseract-OCR\tessdata",
        r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
        str(Path.home() / "AppData" / "Local" / "Programs" / "Tesseract-OCR" / "tessdata"),
    ]

    tessdata_dir = next((path for path in candidate_data_dirs if path and Path(path).exists()), "")
    if tessdata_dir:
        os.environ["TESSDATA_PREFIX"] = tessdata_dir
    else:
        print("[warn] Tesseract tessdata directory was not found. Set TESSDATA_PREFIX.")
        return requested_lang

    available_langs = {
        traineddata.stem
        for traineddata in Path(tessdata_dir).glob("*.traineddata")
    }
    requested_langs = [lang for lang in requested_lang.split("+") if lang]
    missing_langs = [lang for lang in requested_langs if lang not in available_langs]
    usable_langs = [lang for lang in requested_langs if lang in available_langs]

    if missing_langs:
        print(f"[warn] Missing Tesseract language packs: {', '.join(missing_langs)}")
        if usable_langs:
            fallback_lang = "+".join(usable_langs)
            print(f"[warn] Falling back OCR language from '{requested_lang}' to '{fallback_lang}'.")
            return fallback_lang

    return requested_lang


OCR_LANG = _configure_tesseract()
# ================================================================


def _as_doc(text: str, source: str, **meta) -> Document:
    return Document(page_content=text, metadata={"source": source, **meta})


def _dedup_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"), hash(d.page_content))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


# ──────────────────────────────
# 1) Text extraction: PyMuPDF4LLM (Markdown)
# ──────────────────────────────
def extract_text_with_pymupdf4llm(pdf_path: str) -> List[Document]:
    try:
        md = pymupdf4llm.to_markdown(pdf_path)
        if md and md.strip():
            return [_as_doc(md, pdf_path, parser="pymupdf4llm", type="markdown")]
        return []
    except Exception as e:
        print(f"[warn] PyMuPDF4LLM failed: {pdf_path} -> {e}")
        return []


# ──────────────────────────────
# 2) OCR embedded images in each page.
# ──────────────────────────────
def ocr_images_in_pdf(pdf_path: str) -> List[Document]:
    docs: List[Document] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[warn] Failed to open PDF with PyMuPDF: {pdf_path} -> {e}")
        return docs

    for page_index in range(len(doc)):
        try:
            page = doc[page_index]
            images = page.get_images(full=True)
        except Exception as e:
            print(f"[warn] Failed to read image list (page {page_index}): {e}")
            continue

        if not images:
            continue

        for img_idx, img in enumerate(images):
            try:
                xref = img[0]
                info = doc.extract_image(xref)
                img_bytes = info.get("image", None)
                if not img_bytes:
                    continue
                w = info.get("width", 0)
                h = info.get("height", 0)
                if w < OCR_MIN_W or h < OCR_MIN_H:
                    # Skip tiny icons and logos.
                    continue

                pil = Image.open(io.BytesIO(img_bytes))
                text = pytesseract.image_to_string(pil, lang=OCR_LANG) or ""
                text = text.strip()
                if text:
                    docs.append(_as_doc(
                        text=text,
                        source=pdf_path,
                        parser="ocr-image",
                        page=page_index,
                        image_index=img_idx,
                        width=w,
                        height=h,
                        type="ocr"
                    ))
            except Exception as e:
                print(f"[warn] Embedded image OCR failed (page {page_index}, img {img_idx}): {e}")
                continue

    return docs


# ──────────────────────────────
# 3) OCR fallback: render full pages and OCR them.
# ──────────────────────────────
def ocr_pages_rendered(pdf_path: str) -> List[Document]:
    if not PAGE_OCR_ENABLE:
        return []
    docs: List[Document] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[warn] Failed to open PDF with PyMuPDF: {pdf_path} -> {e}")
        return docs

    mat = fitz.Matrix(PAGE_OCR_SCALE, PAGE_OCR_SCALE)
    for page_index in range(len(doc)):
        try:
            pix = doc[page_index].get_pixmap(matrix=mat)
            mode = "RGBA" if pix.alpha else "RGB"
            pil = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(pil, lang=OCR_LANG) or ""
            text = text.strip()
            if text:
                docs.append(_as_doc(
                    text=text,
                    source=pdf_path,
                    parser="ocr-page",
                    page=page_index,
                    width=pix.width,
                    height=pix.height,
                    type="ocr"
                ))
        except Exception as e:
            print(f"[warn] Rendered page OCR failed (page {page_index}): {e}")
            continue
    return docs


# ──────────────────────────────
# Parse one PDF: text extraction + embedded image OCR + optional rendered-page OCR fallback.
# ──────────────────────────────
def parse_pdf(pdf_path: str) -> List[Document]:
    docs: List[Document] = []

    # 1) Text (Markdown)
    md_docs = extract_text_with_pymupdf4llm(pdf_path)
    docs.extend(md_docs)

    # 2) Embedded image OCR
    img_ocr_docs = ocr_images_in_pdf(pdf_path)
    docs.extend(img_ocr_docs)

    # 3) Rendered page OCR when extracted text is too sparse.
    if len("".join(d.page_content for d in docs)) < 500:
        page_ocr_docs = ocr_pages_rendered(pdf_path)
        docs.extend(page_ocr_docs)

    return _dedup_docs(docs)


# ──────────────────────────────
# Directory -> parse -> chunk -> embed -> save to FAISS.
# ──────────────────────────────
def load_documents(source_dir: str = "data") -> List[Document]:
    pdf_paths = list(Path(source_dir).rglob("*.pdf"))
    all_docs: List[Document] = []
    if not pdf_paths:
        print(f"[warn] No PDF files found: {source_dir}")
        return all_docs

    for p in pdf_paths:
        try:
            parsed = parse_pdf(str(p))
            if not parsed:
                print(f"[warn] No parsed content: {p}")
            all_docs.extend(parsed)
        except Exception as e:
            print(f"[warn] Failed to parse: {p} -> {e}")

    return all_docs


def create_vector_store(source_dir: str = "data") -> None:
    """
    Pipeline:
    - Parse PDFs -> split chunks -> OpenAI embeddings -> save FAISS index.
    """
    raw_docs = load_documents(source_dir)
    if not raw_docs:
        raise RuntimeError("No documents found for indexing. Check source_dir.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=getattr(config, "CHUNK_SIZE", 1000),
        chunk_overlap=getattr(config, "CHUNK_OVERLAP", 150),
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"[ok] Documents: {len(raw_docs)} -> chunks: {len(chunks)}")

    embeddings = OpenAIEmbeddings(
        model=getattr(config, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        dimensions=getattr(config, "OPENAI_EMBEDDING_DIMENSIONS", None),
        chunk_size=getattr(config, "EMBED_BATCH_SIZE", 128),
    )

    vs = FAISS.from_documents(chunks, embeddings)
    save_path = getattr(config, "VECTOR_STORE_PATH", getattr(config, "VECTOR_DB_PATH", "vector_store"))
    Path(save_path).mkdir(parents=True, exist_ok=True)
    vs.save_local(save_path)
    print(f"[ok] FAISS index saved: {save_path}")


if __name__ == "__main__":
    create_vector_store()

