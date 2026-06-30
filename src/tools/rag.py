from typing import List

import torch
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config


_USE_RERANKER: bool = bool(getattr(config, "USE_RERANKER", False))
_RERANKER_MODEL_NAME: str = getattr(
    config,
    "RERANKER_MODEL_NAME",
    getattr(config, "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
)

_rerank_tokenizer = None
_rerank_model = None

if _USE_RERANKER:
    try:
        print(f"[rag] Loading reranker: {_RERANKER_MODEL_NAME}")
        _rerank_tokenizer = AutoTokenizer.from_pretrained(_RERANKER_MODEL_NAME)
        _rerank_model = AutoModelForSequenceClassification.from_pretrained(_RERANKER_MODEL_NAME)
        print("[rag] Reranker loaded.")
    except Exception as e:
        print(f"[rag] Failed to load reranker: {e}")
        _USE_RERANKER = False
        _rerank_tokenizer = None
        _rerank_model = None


def _load_faiss() -> FAISS:
    emb = OpenAIEmbeddings(
        model=getattr(config, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        dimensions=getattr(config, "OPENAI_EMBEDDING_DIMENSIONS", None),
    )
    vs_path = getattr(config, "VECTOR_STORE_PATH", getattr(config, "VECTOR_DB_PATH", "vector_store"))
    return FAISS.load_local(vs_path, emb, allow_dangerous_deserialization=True)


def _dedup(docs: List[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"), hash(d.page_content))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _rerank(query: str, docs: List[Document], out_k: int) -> List[Document]:
    if not (_USE_RERANKER and _rerank_model and _rerank_tokenizer and docs):
        return docs[:out_k]
    pairs = [(query, d.page_content) for d in docs]
    inputs = _rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        scores = _rerank_model(**inputs).logits.squeeze().tolist()
    ranked = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return ranked[:out_k]


@tool
def vector_store_rag_search(query: str, top_k: int = None, rerank_k: int = None) -> List[Document]:
    """
    Search local FAISS vector store for documents similar to query.
    """
    print(f"[tool] Running RAG tool: '{query}'")
    if not query:
        return []

    try:
        vs = _load_faiss()
        cfg_topk = int(getattr(config, "TOP_K_PER_QUERY", 5))
        out_k = int(rerank_k or cfg_topk)
        fetch_k = int(top_k or max(out_k * 2, cfg_topk * 2))

        retriever = vs.as_retriever(search_kwargs={"k": fetch_k})
        candidates: List[Document] = retriever.invoke(query)

        uniq = _dedup(candidates)
        ranked = _rerank(query, uniq, out_k=out_k)
        return ranked[:out_k]
    except Exception as e:
        print(f"[error] RAG tool error: {e}")
        return []
