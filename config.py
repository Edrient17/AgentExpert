# config.py

import os
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- LLM Model Names ---
LLM_MODEL_TEAM1 = "gpt-4o-mini"
LLM_MODEL_TEAM2_EVAL = "gpt-4o-mini"
LLM_MODEL_TEAM3 = "gpt-4o"
LLM_MODEL_SUPER_ROUTER = "gpt-4o-mini"

# --- Embedding & Reranker Model Names ---
EMBEDDING_MODEL = "jhgan/ko-sbert-nli"
RERANKER_MODEL = "BAAI/bge-reranker-large"

# --- Vector Store Path ---
VECTOR_STORE_PATH = "vector_store/"

# --- API Keys ---
# 필요한 API 키가 설정되었는지 확인
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
if not os.getenv("SERPAPI_API_KEY"):
    print("SERPAPI_API_KEY가 설정되지 않았습니다.")
