# main_graph.py
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict

# === 상태 정의 ===
class AgentState(TypedDict):
    user_input: str
    q_validity: bool
    q_en_transformed: str

    # Team1 산출물 (추가)
    rag_queries: list[str]        # 후보 쿼리들
    rag_query: str                # supervisor_team1이 최종 선택한 단일 쿼리
    rag_query_scores: list[float] # 후보 쿼리별 점수(길이 = rag_queries)
    output_format: list[str]      # ["type","language"] e.g., ["qa","ko"]

    # 후속 단계
    rag_docs: list
    web_docs: list
    generated_answer: str

    # 에러/제어
    error_message: str
    status: dict
    next_node: str

# === 유틸: Supervisor 래퍼 ===
def wrap_supervisor(team_name: str, supervisor_func):
    def wrapped(state: dict) -> dict:
        return supervisor_func(state)
    return RunnableLambda(wrapped)

# === 에이전트 및 슈퍼바이저 불러오기 ===
from agents.supervisor_team1 import supervisor_team1
from agents.supervisor_team2 import supervisor_team2
from agents.supervisor_team3 import supervisor_team3
from agents.chief_supervisor import chief_supervisor_router

# === 그래프 빌드 ===
builder = StateGraph(AgentState)
builder.set_entry_point("chief_supervisor")

# === 노드 등록 ===
builder.add_node("chief_supervisor", RunnableLambda(chief_supervisor_router))

# === Team 1 ===
builder.add_node("team1_supervisor", wrap_supervisor("team1", supervisor_team1))
builder.add_edge("team1_supervisor", "chief_supervisor")

# === Team 2 ===
builder.add_node("team2_supervisor", wrap_supervisor("team2", supervisor_team2))
builder.add_edge("team2_supervisor", "chief_supervisor")

# === Team 3 ===
builder.add_node("team3_supervisor", wrap_supervisor("team3", supervisor_team3))
builder.add_edge("team3_supervisor", "chief_supervisor")

# === 조건 분기: 모든 제어는 chief_supervisor_router에서 처리 ===
builder.add_conditional_edges(
    "chief_supervisor",
    lambda state: state["next_node"],
    {
        "team1_supervisor": "team1_supervisor",
        "team2_supervisor": "team2_supervisor",
        "team3_supervisor": "team3_supervisor",
        "end": END,
    },
)

# === 컴파일 ===
graph = builder.compile()

if __name__ == "__main__":
    print("\n📌 LangGraph 구조 (ASCII View):")
    graph.get_graph().print_ascii()
