# utility_tools.py

import os
import requests
import torch
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config

# --- Reranker 모델 전역 로드 ---
print("ReRanker 모델을 로드합니다...")
try:
    rerank_tokenizer = AutoTokenizer.from_pretrained(config.RERANKER_MODEL)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(config.RERANKER_MODEL)
    print("ReRanker 모델 로드 완료.")
except Exception as e:
    print(f"ReRanker 모델 로드 실패: {e}")
    rerank_model = None

def format_docs(docs: List[Document], max_chars: int = 15000) -> str:
    """
    LangChain Document 객체 리스트를 LLM 프롬프트에 삽입하기 좋은 단일 문자열로 결합합니다.
    """
    contents = [doc.page_content.strip() for doc in docs if isinstance(doc, Document) and doc.page_content]
    if not contents:
        return "[NO CONTENT]"

    joined_content = "\n\n---\n\n".join(contents)
    return joined_content[:max_chars] + "\n\n...[내용 일부 생략]..." if len(joined_content) > max_chars else joined_content

# --- Tool 1: Vector Store RAG 검색 ---
@tool
def vector_store_rag_search(query: str, top_k: int = 10, rerank_k: int = 3) -> List[Document]:
    """
    사용자의 질문(query)을 받아 로컬 Vector Store(FAISS)에서 관련 문서를 검색하고,
    정확도를 높이기 위해 리랭킹(reranking)을 수행한 후 상위 문서를 반환합니다.
    내부 지식 베이스에서 정보를 찾아야 할 때 사용합니다.
    """
    print(f"🛠️ RAG Tool 실행: '{query}'")
    if not query: return []
    try:
        embedding = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        db = FAISS.load_local(config.VECTOR_STORE_PATH, embedding, allow_dangerous_deserialization=True)
        candidate_docs = db.similarity_search(query, k=top_k)

        if rerank_model and candidate_docs:
            pairs = [(query, doc.page_content) for doc in candidate_docs]
            inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad():
                scores = rerank_model(**inputs).logits.squeeze().tolist()
            scored_docs = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
            return [doc for score, doc in scored_docs[:rerank_k]]
        return candidate_docs[:rerank_k]
    except Exception as e:
        print(f"❌ RAG Tool 실행 중 오류 발생: {e}")
        return []

# --- Tool 2: OpenAI 웹 검색 ---

# --- Pydantic 스키마 (웹 검색 결과 구조화) ---
class SearchResult(BaseModel):
    """A single search result with its title, URL, and comprehensive summary."""
    title: str = Field(description="The title of the search result.")
    url: str = Field(description="The URL of the search result.")
    summary: str = Field(description="A detailed and comprehensive summary of the search result's content, answering the user's query.")

class SearchResults(BaseModel):
    """A list of search results."""
    results: List[SearchResult]

@tool
def deep_research_web_search(query: str, max_results: int = 3) -> List[Document]:
    """
    사용자의 질문(query)에 대해 gpt-4.1 모델을 사용하여 심층적인 웹 리서치를 수행하고,
    그 결과를 구조화된 Document 객체 리스트로 반환합니다.
    최신 정보나 외부의 깊이 있는 지식이 필요할 때 사용합니다.
    """
    print(f"🛠️ Deep Research Tool 실행: '{query}'")
    if not query: return []

    try:
        # LLM 및 프롬프트 설정
        llm = ChatOpenAI(model="gpt-4.1", temperature=0)
        structured_llm = llm.with_structured_output(SearchResults)
        
        prompt = PromptTemplate.from_template(
            """You are an expert web researcher. Your task is to conduct a thorough and objective web search to answer the user's query.
            
            Please find {max_results} distinct, detailed, and non-overlapping results that directly address the following query: '{query}'
            
            For each result, provide a clear title, the source URL, and a comprehensive summary. Ensure the summary is detailed enough to be useful on its own."""
        )

        # 체인 실행
        chain = prompt | structured_llm
        response = chain.invoke({"query": query, "max_results": max_results})

        # 결과를 Document 객체로 변환
        docs = []
        if response and response.results:
            for result in response.results:
                doc = Document(
                    page_content=f"Title: {result.title}\nSummary: {result.summary}",
                    metadata={"source": result.url, "title": result.title}
                )
                docs.append(doc)
        
        return docs

    except Exception as e:
        print(f"❌ Deep Research Tool 실행 중 오류 발생: {e}")
        return []