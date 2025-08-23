# utility_tools.py

import os
import torch
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config

# =========================================================
# 공통 유틸
# =========================================================

def format_docs(docs: List[Document], max_chars: int = 15000) -> str:
    """
    LangChain Document 리스트를 프롬프트에 넣기 쉬운 하나의 문자열로 결합.
    """
    contents = [d.page_content.strip() for d in docs if isinstance(d, Document) and d.page_content]
    if not contents:
        return "[NO CONTENT]"
    joined = "\n\n---\n\n".join(contents)
    return (joined[:max_chars] + "\n\n...[내용 일부 생략]...") if len(joined) > max_chars else joined


# =========================================================
# Reranker 전역 로드 (옵션)
# =========================================================

_USE_RERANKER: bool = bool(getattr(config, "USE_RERANKER", False))
_RERANKER_MODEL_NAME: str = getattr(
    config,
    "RERANKER_MODEL_NAME",
    getattr(config, "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
)

_rerank_tokenizer = None
_rerank_model = None

if _USE_RERANKER:
    try:
        print(f"[utility_tools] 리랭커 로드: {_RERANKER_MODEL_NAME}")
        _rerank_tokenizer = AutoTokenizer.from_pretrained(_RERANKER_MODEL_NAME)
        _rerank_model = AutoModelForSequenceClassification.from_pretrained(_RERANKER_MODEL_NAME)
        print("[utility_tools] 리랭커 로드 완료")
    except Exception as e:
        print(f"[utility_tools] 리랭커 로드 실패: {e}")
        _USE_RERANKER = False
        _rerank_tokenizer = None
        _rerank_model = None


# =========================================================
# Vector Store (FAISS) 로드
# =========================================================

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


# =========================================================
# Tool 1: Vector Store RAG 검색
#  - OpenAI 임베딩(다국어) 사용
#  - (옵션) dual_queries 병행 검색
# =========================================================

@tool
def vector_store_rag_search(
    query: str,
    top_k: int = None,
    rerank_k: int = None
) -> List[Document]:
    """
    로컬 FAISS 벡터스토어에서 유사 문서를 검색합니다.
    - query: 기본(주로 영어) 쿼리
    - top_k: 1차 후보 개수(미지정 시 config.TOP_K_PER_QUERY*2 정도로 내부 조정)
    - rerank_k: 최종 반환 개수(미지정 시 config.TOP_K_PER_QUERY)
    """
    print(f"🛠️ RAG Tool 실행: '{query}'")
    if not query:
        return []

    try:
        vs = _load_faiss()

        cfg_topk = int(getattr(config, "TOP_K_PER_QUERY", 5))
        out_k = int(rerank_k or cfg_topk)
        fetch_k = int(top_k or max(out_k * 2, cfg_topk * 2))

        retriever = vs.as_retriever(search_kwargs={"k": fetch_k})
        candidates: List[Document] = retriever.invoke(query)

        # 중복 제거 후 (옵션) 리랭킹
        uniq = _dedup(candidates)
        ranked = _rerank(query, uniq, out_k=out_k)

        return ranked[:out_k]

    except Exception as e:
        print(f"❌ RAG Tool 실행 오류: {e}")
        return []


# =========================================================
# Tool 2: (LLM 기반) 심층 웹 리서치
# =========================================================

class SearchResult(BaseModel):
    title: str = Field(description="The title of the search result.")
    url: str = Field(description="The URL of the search result.")
    summary: str = Field(description="A detailed, objective summary addressing the user's query.")

class SearchResults(BaseModel):
    results: List[SearchResult]

@tool
def deep_research_web_search(query: str, max_results: int = 3) -> List[Document]:
    """
    gpt-4.1 등 LLM을 이용해 구조화된 웹 리서치 요약을 생성합니다.
    (실제 네비게이션/브라우징 없이 요약 기반)
    """
    print(f"🛠️ Deep Research Tool 실행: '{query}'")
    if not query:
        return []

    try:
        llm = ChatOpenAI(model=getattr(config, "LLM_MODEL_WEB", "gpt-4.1"), temperature=0)
        structured_llm = llm.with_structured_output(SearchResults)

        prompt = PromptTemplate.from_template(
"""You are an expert, objective web researcher working specifically on Samsung-related questions.
Assume that most incoming queries are about Samsung products, services, specifications, policies, or company news.

Goal:
Provide {max_results} distinct, non-overlapping results that directly address the user’s query,
prioritizing Samsung-owned official sources while allowing reputable third-party sources only if official coverage is insufficient.

Ranking & Source Policy:
- Rank Samsung official domains first (samsung.com, semiconductor.samsung.com, news.samsung.com, images.samsung.com, developer.samsung.com, research.samsung.com).
- If official coverage is insufficient, you MAY include reputable third-party sources (e.g., major tech media, standards bodies, vendor whitepapers),
  but keep the summaries factual and clearly avoid speculation.

Requirements:
- Each result must include:
  (1) a clear, concise title,
  (2) a comprehensive, factual summary in Korean (useful on its own, no marketing fluff; if third-party, reflect that neutrally),
  (3) the canonical, direct https URL.
- Prefer diverse domains/perspectives; avoid near-duplicates.
- Do NOT fabricate facts or URLs. If a crucial detail is unknown, explicitly state it in the summary.
- Be precise and concrete: include key numbers, model names, SKUs, standards, or dates when available.

Query: '{query}'

Return ONLY a structured object that matches the given schema (no extra text).
"""
        )

        chain = prompt | structured_llm
        response: SearchResults = chain.invoke({"query": query, "max_results": max_results})

        docs: List[Document] = []
        if response and response.results:
            for r in response.results:
                docs.append(
                    Document(
                        page_content=f"Title: {r.title}\nSummary: {r.summary}",
                        metadata={"source": r.url, "title": r.title}
                    )
                )
        return docs

    except Exception as e:
        print(f"❌ Deep Research Tool 실행 오류: {e}")
        return []
