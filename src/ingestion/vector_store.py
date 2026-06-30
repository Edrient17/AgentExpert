from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from src.ingestion.pdf_parser import load_documents


def create_vector_store(source_dir: str = "data") -> None:
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

    vector_store = FAISS.from_documents(chunks, embeddings)
    save_path = getattr(config, "VECTOR_STORE_PATH", getattr(config, "VECTOR_DB_PATH", "vector_store"))
    Path(save_path).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(save_path)
    print(f"[ok] FAISS index saved: {save_path}")
