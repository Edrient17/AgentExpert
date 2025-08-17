# graphs/team2_graph.py

from langgraph.graph import StateGraph, END

from state import GlobalState
from agents.team2_agents import rag_search, web_search, evaluate_documents

# 각 검색 단계별 재시도 횟수 설정
MAX_RETRIES = 2

def retrieval_router(state: GlobalState) -> str:
    """
    RAG/웹 검색 및 평가 결과를 바탕으로 재시도, 대체(Fallback), 성공, 실패를 결정합니다.
    """

    retries = state.get("team2_retries", 0)
    web_search_attempted = state.get("web_docs") is not None

    # 1. 평가 통과: 현재 단계(RAG 또는 Web)가 성공했으므로 전체 성공으로 종료
    status = state.get("status", {}).get("team2")
    if status == "pass":
        print("✅ Team 2 통과.")
        state["team2_retries"] = 0
        return END

    # 2. 평가 실패: 재시도 또는 다음 단계로 전환
    if not web_search_attempted: # RAG 단계
        if retries < MAX_RETRIES:
            print(f"🔁 RAG 검색: {retries + 1} 번째 시도")
            return "rag_search"
        else:
            print("🔁 RAG 최종 실패. 웹 검색으로 대체 작동(Fallback)합니다.")
            return "web_search"
    else: # 웹 검색 단계
        if retries < MAX_RETRIES:
            print(f"🔁 WEB 검색: {retries + 1} 번째 시도")
            return "web_search"
        else:
            print(f"❌ Team 2 최종 실패 (재시도 {MAX_RETRIES}회 초과).")
            return END

def create_team2_graph():
    """
    Team 2의 서브그래프를 생성하고 컴파일합니다.
    - 재시도 로직이 포함된 RAG 우선, 웹 검색 대체(Fallback) 워크플로우
    """
    builder = StateGraph(GlobalState)

    # 노드 추가
    builder.add_node("rag_search", rag_search)
    builder.add_node("web_search", web_search)
    builder.add_node("evaluate_documents", evaluate_documents)

    # 엣지 연결
    builder.set_entry_point("rag_search")
    builder.add_edge("rag_search", "evaluate_documents")
    builder.add_edge("web_search", "evaluate_documents")
    
    # 조건부 엣지 추가
    builder.add_conditional_edges(
        "evaluate_documents",
        retrieval_router,
        {
            "rag_search": "rag_search",
            "web_search": "web_search",
            END: END
        }
    )

    # 그래프 컴파일
    return builder.compile()