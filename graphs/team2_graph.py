# graphs/team2_graph.py

from langgraph.graph import StateGraph, END

from state import AgentState
from agents.team2_agents import rag_search, web_search, evaluate_documents



def create_team2_graph():
    """
    Team 2의 서브그래프를 생성하고 컴파일합니다.
    """
    builder = StateGraph(AgentState)

    builder.add_node("rag_search", rag_search)
    builder.add_node("web_search", web_search)
    builder.add_node("evaluate_documents", evaluate_documents)

    builder.set_entry_point("rag_search")
    builder.add_edge("rag_search", "evaluate_documents")
    builder.add_edge("web_search", "evaluate_documents")
    
    def route_after_evaluation(state: AgentState) -> str:
        """
        evaluate_documents 노드가 보낸 신호에 따라 다음 단계를 결정합니다.
        """
        last_message = state['messages'][-1]
        decision = last_message.content

        print(f"🚦 라우터: Team 2 '{decision}' 신호 수신.")
        
        if decision == "retry_rag":
            return "rag_search"
        elif decision == "fallback_to_web":
            return "web_search"
        elif decision == "retry_web":
            return "web_search"
        else:
            return END

    builder.add_conditional_edges(
        "evaluate_documents",
        route_after_evaluation,
        {
            "rag_search": "rag_search",
            "web_search": "web_search",
            END: END
        }
    )

    return builder.compile()