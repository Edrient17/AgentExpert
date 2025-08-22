from __future__ import annotations

from typing import TypedDict, Annotated, Optional, List, Literal
from operator import add as list_concat

from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    # 대화 이력 (LangGraph 표준: 메시지는 add_messages로 축적)
    messages: Annotated[List[AnyMessage], add_messages]

    # 매니저 라우팅/피드백
    next_team_to_call: Optional[Literal["team1", "team2", "team3", "end"]]
    manager_feedback: Optional[str]

    # ──────────────────────────────
    # 접근성 향상: 직접 접근 키
    # ──────────────────────────────
    # 원문 사용자 입력(ko/en) — Team1에서 세팅
    user_input: Optional[str]

    # Team1이 만든 영어 정규화 질의(생성/정규화 결과)
    q_en_transformed: Optional[str]

    # Team1 Evaluator가 고른 최종 RAG 쿼리
    best_rag_query: Optional[str]

    # Team2가 누적/평가해 모으는 문서 버킷
    # (하위 노드 반복 호출 동안 계속 append되어야 하므로 list_concat로 누적)
    rag_docs: Annotated[List[Document], list_concat]
    web_docs: Annotated[List[Document], list_concat]

    # 재시도/루프 카운터
    team1_retries: int
    team2_retries: int
    team3_retries: int
    global_loop_count: int