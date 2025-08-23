# config.py
import os
from typing import Optional
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# -----------------------------
# LLM 모델 설정
# -----------------------------
LLM_MODEL_TEAM1: str = "gpt-4.1"
LLM_MODEL_TEAM2_EVAL: str = "gpt-4.1"
LLM_MODEL_TEAM3: str = "gpt-4.1" 
LLM_MODEL_SUPER_ROUTER: str = "gpt-4.1"
LLM_MODEL_WEB: str = "gpt-4.1"

# Team3 생성 파라미터
TEAM3_TEMPERATURE: float = 0

# -----------------------------
# 벡터 스토어 / 임베딩
# -----------------------------
# 인덱스 저장 경로
VECTOR_STORE_PATH: str = "vector_store/"
# (하위 호환용) 일부 코드가 VECTOR_DB_PATH를 참조할 수 있어 동치로 둡니다.
VECTOR_DB_PATH: str = VECTOR_STORE_PATH

# OpenAI 임베딩 모델(다국어)
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
# dimensions=None 이면 기본(3072). 1024/256 등으로 줄이면 인덱스/비용 절감
OPENAI_EMBEDDING_DIMENSIONS: Optional[int] = None

# 청크 분할
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150

# 검색 파라미터
TOP_K_PER_QUERY: int = 5

# (선택) 리랭커 사용: utility_tools에서 참고
USE_RERANKER: bool = False
RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"

# 차트 산출물 저장 경로/옵션
CHART_OUTPUT_DIR: str = "artifacts/charts"
CHART_DPI: int = 150


# -----------------------------
# 제어 플로우(재시도/루프)
# -----------------------------
MAX_RETRIES_TEAM1: int = 2
MAX_RETRIES_TEAM2: int = 4
MAX_RETRIES_TEAM3: int = 2
MAX_GLOBAL_LOOPS: int = 2

# -----------------------------
# 필수 키 체크
# -----------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
