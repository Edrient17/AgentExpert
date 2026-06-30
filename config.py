# config.py
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env.
load_dotenv()

# -----------------------------
# LLM model settings
# -----------------------------
LLM_MODEL_TEAM1: str = "gpt-4.1"
LLM_MODEL_TEAM2_EVAL: str = "gpt-4.1"
LLM_MODEL_TEAM3_GEN: str = "gpt-4.1"
LLM_MODEL_TEAM3_EVAL: str = "gpt-4.1"
LLM_MODEL_SUPER_ROUTER: str = "gpt-4.1"
LLM_MODEL_WEB: str = "gpt-4.1"

# Team3 generation parameters
TEAM3_TEMPERATURE: float = 0

# -----------------------------
# Vector store / embeddings
# -----------------------------
# Index storage path
VECTOR_STORE_PATH: str = "vector_store/"
# Backward compatibility: some code may still reference VECTOR_DB_PATH.
VECTOR_DB_PATH: str = VECTOR_STORE_PATH

# OpenAI embedding model
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
# dimensions=None uses the model default. Lower dimensions can reduce index size/cost.
OPENAI_EMBEDDING_DIMENSIONS: Optional[int] = None

# Chunking
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150

# OCR settings
OCR_LANG: str = os.getenv("TESSERACT_LANG", "eng")
TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "")
TESSDATA_PREFIX: str = os.getenv("TESSDATA_PREFIX", "")

# Retrieval parameters
TOP_K_PER_QUERY: int = 5

# Optional reranker settings
USE_RERANKER: bool = False
RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"

# -----------------------------
# Control flow settings
# -----------------------------
MAX_RETRIES_TEAM1: int = 2
MAX_RETRIES_TEAM2: int = 4
MAX_RETRIES_TEAM3: int = 2
MAX_GLOBAL_LOOPS: int = 2

# -----------------------------
# Required key check
# -----------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not configured.")
