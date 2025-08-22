# ingest_data.py
"""
OpenAI text-embedding-3-large 기반 인덱싱 파이프라인
- PDF: PyMuPDFLoader
- 텍스트(.txt/.md): 직접 로드
- 이미지(.png/.jpg/.jpeg): pytesseract OCR (kor+eng 기본)
- 청크 분할 후 FAISS로 저장
"""

import os
from pathlib import Path
from typing import List

import pytesseract
from PIL import Image

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

import config


def _iter_files(root: str) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"소스 디렉토리가 존재하지 않습니다: {root}")
    files: List[Path] = []
    for p in root_path.rglob("*"):
        if p.is_file():
            files.append(p)
    return files


def _load_pdf(path: Path) -> List[Document]:
    try:
        loader = PyMuPDFLoader(str(path))
        return loader.load()
    except Exception as e:
        print(f"[warn] PDF 로드 실패: {path} -> {e}")
        return []


def _load_text(path: Path) -> List[Document]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": str(path), "type": "text"})]
    except Exception as e:
        print(f"[warn] 텍스트 로드 실패: {path} -> {e}")
        return []


def _ocr_image(path: Path) -> List[Document]:
    """
    Tesseract가 설치되어 있어야 합니다.
    - 환경변수 TESSERACT_CMD 로 경로 지정 가능.
    - 기본 언어: kor+eng (없으면 eng로 폴백)
    """
    t_cmd = os.getenv("TESSERACT_CMD")
    if t_cmd:
        pytesseract.pytesseract.tesseract_cmd = t_cmd

    lang = os.getenv("TESSERACT_LANG", "kor+eng")
    try:
        img = Image.open(str(path))
        text = pytesseract.image_to_string(img, lang=lang)
        text = text.strip()
        if not text:
            print(f"[warn] OCR 결과가 비었습니다: {path}")
            return []
        return [Document(page_content=text, metadata={"source": str(path), "type": "image-ocr"})]
    except Exception as e:
        print(f"[warn] OCR 실패: {path} -> {e}")
        return []


def load_documents(source_dir: str = "data") -> List[Document]:
    docs: List[Document] = []
    for fp in _iter_files(source_dir):
        suffix = fp.suffix.lower()
        if suffix == ".pdf":
            docs.extend(_load_pdf(fp))
        elif suffix in [".txt", ".md"]:
            docs.extend(_load_text(fp))
        elif suffix in [".png", ".jpg", ".jpeg"]:
            docs.extend(_ocr_image(fp))
        else:
            # 기타 포맷은 건너뜀
            pass
    return docs


def create_vector_store(source_dir: str = "data") -> None:
    """
    OpenAI text-embedding-3-large로 문서 임베딩 후 FAISS에 저장합니다.
    """
    # 1) 문서 로드
    raw_docs = load_documents(source_dir)
    if not raw_docs:
        raise RuntimeError("인덱싱할 문서를 찾지 못했습니다. source_dir을 확인하세요.")

    # 2) 청크 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=getattr(config, "CHUNK_SIZE", 1000),
        chunk_overlap=getattr(config, "CHUNK_OVERLAP", 150),
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"[ok] 문서 {len(raw_docs)}개 → 청크 {len(chunks)}개")

    # 3) 임베딩 (OpenAI)
    embeddings = OpenAIEmbeddings(
        model=getattr(config, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        dimensions=getattr(config, "OPENAI_EMBEDDING_DIMENSIONS", None),
        chunk_size=getattr(config, "EMBED_BATCH_SIZE", 128),
    )

    # 4) 벡터스토어 생성 및 저장
    vs = FAISS.from_documents(chunks, embeddings)
    save_path = getattr(config, "VECTOR_STORE_PATH", "vector_store/")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    vs.save_local(save_path)
    print(f"[ok] FAISS 벡터 저장소 저장 완료: {save_path}")


if __name__ == "__main__":
    create_vector_store()
