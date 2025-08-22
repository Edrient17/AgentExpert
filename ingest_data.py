# ingest_data.py
"""
PDF 인덱싱 파이프라인 (PyMuPDF4LLM + OCR)
- 1) PyMuPDF4LLM: 본문을 Markdown으로 변환
- 2) OCR(이미지): 페이지 내 '내장 이미지'를 추출해 텍스트 보강
- 3) OCR(페이지 렌더 폴백): 스캔 PDF 등에서 텍스트가 부족하면 페이지 자체를 렌더해 추가 OCR
- 결과를 LangChain Document로 정규화 → 청크 → OpenAI 임베딩 → FAISS 저장
"""

from __future__ import annotations

import os
import io
from pathlib import Path
from typing import List, Optional

import fitz               # PyMuPDF (이미지 추출/페이지 렌더)
import pymupdf4llm        # PDF → Markdown
from PIL import Image
import pytesseract

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import config

# ========= 조정 파라미터(필요 시 config.py에 옮겨 사용 가능) =========
OCR_LANG = getattr(config, "OCR_LANG", os.getenv("TESSERACT_LANG", "kor+eng"))
OCR_MIN_W = getattr(config, "OCR_MIN_W", 200)          # OCR 대상 최소 이미지 가로
OCR_MIN_H = getattr(config, "OCR_MIN_H", 200)          # OCR 대상 최소 이미지 세로
PAGE_OCR_ENABLE = getattr(config, "PAGE_OCR_ENABLE", True)   # 페이지 렌더 OCR 폴백 사용
PAGE_OCR_SCALE = getattr(config, "PAGE_OCR_SCALE", 2.0)      # 렌더링 스케일(2.0≈~200~300dpi)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
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
# 1) 본문 추출: PyMuPDF4LLM (Markdown)
# ──────────────────────────────
def extract_text_with_pymupdf4llm(pdf_path: str) -> List[Document]:
    try:
        md = pymupdf4llm.to_markdown(pdf_path)
        if md and md.strip():
            return [_as_doc(md, pdf_path, parser="pymupdf4llm", type="markdown")]
        return []
    except Exception as e:
        print(f"[warn] PyMuPDF4LLM 실패: {pdf_path} -> {e}")
        return []


# ──────────────────────────────
# 2) OCR(이미지): 페이지 내 내장 이미지 대상
# ──────────────────────────────
def ocr_images_in_pdf(pdf_path: str) -> List[Document]:
    docs: List[Document] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[warn] PyMuPDF open 실패: {pdf_path} -> {e}")
        return docs

    for page_index in range(len(doc)):
        try:
            page = doc[page_index]
            images = page.get_images(full=True)
        except Exception as e:
            print(f"[warn] 이미지 목록 조회 실패(page {page_index}): {e}")
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
                    # 너무 작은 아이콘/로고는 건너뜀
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
                print(f"[warn] OCR 이미지 추출 실패(page {page_index}, img {img_idx}): {e}")
                continue

    return docs


# ──────────────────────────────
# 3) OCR(폴백): 페이지 렌더링 후 전체 OCR
# ──────────────────────────────
def ocr_pages_rendered(pdf_path: str) -> List[Document]:
    if not PAGE_OCR_ENABLE:
        return []
    docs: List[Document] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[warn] PyMuPDF open 실패: {pdf_path} -> {e}")
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
            print(f"[warn] 페이지 OCR 실패(page {page_index}): {e}")
            continue
    return docs


# ──────────────────────────────
# 단일 PDF 처리: 본문 + 이미지 OCR + (옵션) 페이지 OCR 폴백
# ──────────────────────────────
def parse_pdf(pdf_path: str) -> List[Document]:
    docs: List[Document] = []

    # 1) 본문(Markdown)
    md_docs = extract_text_with_pymupdf4llm(pdf_path)
    docs.extend(md_docs)

    # 2) 이미지 OCR
    img_ocr_docs = ocr_images_in_pdf(pdf_path)
    docs.extend(img_ocr_docs)

    # 3) 페이지 렌더 OCR (본문/이미지 OCR이 너무 빈약할 때 보강)
    if len("".join(d.page_content for d in docs)) < 500:
        page_ocr_docs = ocr_pages_rendered(pdf_path)
        docs.extend(page_ocr_docs)

    return _dedup_docs(docs)


# ──────────────────────────────
# 디렉토리 → 파싱 → 청크 → 임베딩 → FAISS 저장
# ──────────────────────────────
def load_documents(source_dir: str = "data") -> List[Document]:
    pdf_paths = list(Path(source_dir).rglob("*.pdf"))
    all_docs: List[Document] = []
    if not pdf_paths:
        print(f"[warn] PDF 없음: {source_dir}")
        return all_docs

    for p in pdf_paths:
        try:
            parsed = parse_pdf(str(p))
            if not parsed:
                print(f"[warn] 파싱 결과 없음: {p}")
            all_docs.extend(parsed)
        except Exception as e:
            print(f"[warn] 파싱 실패: {p} -> {e}")

    return all_docs


def create_vector_store(source_dir: str = "data") -> None:
    """
    파이프라인:
    - PDF 파싱 → 청크 → OpenAI 임베딩(text-embedding-3-large) → FAISS 저장
    """
    raw_docs = load_documents(source_dir)
    if not raw_docs:
        raise RuntimeError("인덱싱할 문서를 찾지 못했습니다. source_dir을 확인하세요.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=getattr(config, "CHUNK_SIZE", 1000),
        chunk_overlap=getattr(config, "CHUNK_OVERLAP", 150),
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"[ok] 문서 {len(raw_docs)}개 → 청크 {len(chunks)}개")

    embeddings = OpenAIEmbeddings(
        model=getattr(config, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        dimensions=getattr(config, "OPENAI_EMBEDDING_DIMENSIONS", None),
        chunk_size=getattr(config, "EMBED_BATCH_SIZE", 128),
    )

    vs = FAISS.from_documents(chunks, embeddings)
    save_path = getattr(config, "VECTOR_STORE_PATH", getattr(config, "VECTOR_DB_PATH", "vector_store"))
    Path(save_path).mkdir(parents=True, exist_ok=True)
    vs.save_local(save_path)
    print(f"[ok] FAISS 저장 완료: {save_path}")


if __name__ == "__main__":
    create_vector_store()

