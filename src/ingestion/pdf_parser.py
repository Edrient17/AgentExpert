from __future__ import annotations

import io
from pathlib import Path
from typing import List

import fitz
import pymupdf4llm
import pytesseract
from langchain_core.documents import Document
from PIL import Image

import config
from src.ingestion.tesseract import configure_tesseract


OCR_MIN_W = getattr(config, "OCR_MIN_W", 200)
OCR_MIN_H = getattr(config, "OCR_MIN_H", 200)
PAGE_OCR_ENABLE = getattr(config, "PAGE_OCR_ENABLE", True)
PAGE_OCR_SCALE = getattr(config, "PAGE_OCR_SCALE", 2.0)
OCR_LANG = configure_tesseract()


def as_document(text: str, source: str, **meta) -> Document:
    return Document(page_content=text, metadata={"source": source, **meta})


def deduplicate_documents(docs: List[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []
    for doc in docs:
        key = (doc.metadata.get("source"), doc.metadata.get("page"), hash(doc.page_content))
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def extract_text_with_pymupdf4llm(pdf_path: str) -> List[Document]:
    try:
        markdown = pymupdf4llm.to_markdown(pdf_path)
        if markdown and markdown.strip():
            return [as_document(markdown, pdf_path, parser="pymupdf4llm", type="markdown")]
        return []
    except Exception as e:
        print(f"[warn] PyMuPDF4LLM failed: {pdf_path} -> {e}")
        return []


def ocr_images_in_pdf(pdf_path: str) -> List[Document]:
    docs: List[Document] = []
    try:
        pdf = fitz.open(pdf_path)
    except Exception as e:
        print(f"[warn] Failed to open PDF with PyMuPDF: {pdf_path} -> {e}")
        return docs

    for page_index in range(len(pdf)):
        try:
            page = pdf[page_index]
            images = page.get_images(full=True)
        except Exception as e:
            print(f"[warn] Failed to read image list (page {page_index}): {e}")
            continue

        for image_index, image in enumerate(images):
            try:
                xref = image[0]
                info = pdf.extract_image(xref)
                image_bytes = info.get("image")
                if not image_bytes:
                    continue

                width = info.get("width", 0)
                height = info.get("height", 0)
                if width < OCR_MIN_W or height < OCR_MIN_H:
                    continue

                pil_image = Image.open(io.BytesIO(image_bytes))
                text = (pytesseract.image_to_string(pil_image, lang=OCR_LANG) or "").strip()
                if text:
                    docs.append(
                        as_document(
                            text=text,
                            source=pdf_path,
                            parser="ocr-image",
                            page=page_index,
                            image_index=image_index,
                            width=width,
                            height=height,
                            type="ocr",
                        )
                    )
            except Exception as e:
                print(f"[warn] Embedded image OCR failed (page {page_index}, image {image_index}): {e}")

    return docs


def ocr_pages_rendered(pdf_path: str) -> List[Document]:
    if not PAGE_OCR_ENABLE:
        return []

    docs: List[Document] = []
    try:
        pdf = fitz.open(pdf_path)
    except Exception as e:
        print(f"[warn] Failed to open PDF with PyMuPDF: {pdf_path} -> {e}")
        return docs

    matrix = fitz.Matrix(PAGE_OCR_SCALE, PAGE_OCR_SCALE)
    for page_index in range(len(pdf)):
        try:
            pixmap = pdf[page_index].get_pixmap(matrix=matrix)
            mode = "RGBA" if pixmap.alpha else "RGB"
            pil_image = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)
            text = (pytesseract.image_to_string(pil_image, lang=OCR_LANG) or "").strip()
            if text:
                docs.append(
                    as_document(
                        text=text,
                        source=pdf_path,
                        parser="ocr-page",
                        page=page_index,
                        width=pixmap.width,
                        height=pixmap.height,
                        type="ocr",
                    )
                )
        except Exception as e:
            print(f"[warn] Rendered page OCR failed (page {page_index}): {e}")

    return docs


def parse_pdf(pdf_path: str) -> List[Document]:
    docs: List[Document] = []
    docs.extend(extract_text_with_pymupdf4llm(pdf_path))
    docs.extend(ocr_images_in_pdf(pdf_path))

    if len("".join(doc.page_content for doc in docs)) < 500:
        docs.extend(ocr_pages_rendered(pdf_path))

    return deduplicate_documents(docs)


def load_documents(source_dir: str = "data") -> List[Document]:
    pdf_paths = list(Path(source_dir).rglob("*.pdf"))
    all_docs: List[Document] = []
    if not pdf_paths:
        print(f"[warn] No PDF files found: {source_dir}")
        return all_docs

    for path in pdf_paths:
        try:
            parsed_docs = parse_pdf(str(path))
            if not parsed_docs:
                print(f"[warn] No parsed content: {path}")
            all_docs.extend(parsed_docs)
        except Exception as e:
            print(f"[warn] Failed to parse: {path} -> {e}")

    return all_docs
