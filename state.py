# state.py

from typing import TypedDict, List, Dict, Any, Literal
from langchain.schema import Document

class GlobalState(TypedDict):
    """
    전체 워크플로우의 상태를 관리합니다.
    """
    # 입력
    user_input: str

    # Team 1 (질문 분석) 결과
    q_validity: bool
    q_en_transformed: str
    rag_queries: List[str]
    rag_query: str
    output_format: List[str]

    # Team 2 (정보 수집) 결과
    rag_docs: List[Document]
    web_docs: List[Document]

    # Team 3 (답변 생성) 결과
    generated_answer: str

    # 제어 흐름 및 에러 관리
    status: Dict[str, Literal["pass", "fail", "pending"]]
    error_message: str
    
    # 각 팀 서브그래프의 재시도 횟수를 추적
    team1_retries: int
    team2_retries: int
    team3_retries: int