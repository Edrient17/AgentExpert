# team2_rag_agent.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Reranker 모델 로딩 (전역)
rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
rerank_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large")

def rerank_score(query: str, passage: str) -> float:
    inputs = rerank_tokenizer(query, passage, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = rerank_model(**inputs).logits
    return torch.sigmoid(logits[0][0]).item()

def agent_team2_rag_search(state: dict,
                           vector_store_path="vector_store/",
                           model_name="jhgan/ko-sbert-nli",
                           top_k=10,
                           rerank_k=3) -> dict:
    """
    Agent 2: RAG 문서 검색 및 rerank
    """
    query = state.get("rag_query", "")
    new_state = state.copy()

    if not query:
        print("⚠️ RAG 쿼리 없음")
        new_state["rag_docs"] = []
        return new_state

    embedding = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local(vector_store_path, embedding, allow_dangerous_deserialization=True)

    try:
        candidates = db.similarity_search(query, k=top_k)
        scored = [(rerank_score(query, doc.page_content), doc) for doc in candidates]
        reranked = sorted(scored, key=lambda x: x[0], reverse=True)[:rerank_k]
        top_docs = [doc for score, doc in reranked]

        new_state["rag_docs"] = top_docs
        return new_state

    except Exception as e:
        print(f"❌ RAG 검색 실패: {e}")
        new_state["rag_docs"] = []
        return new_state
