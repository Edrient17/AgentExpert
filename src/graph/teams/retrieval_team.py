from langgraph.graph import END, StateGraph

from src.agents.retrieval.evaluator import evaluate_documents
from src.agents.retrieval.rag_worker import rag_search
from src.agents.retrieval.web_worker import web_search
from src.schema.state import AgentState


def create_retrieval_team_graph():
    builder = StateGraph(AgentState)

    builder.add_node("rag_search", rag_search)
    builder.add_node("web_search", web_search)
    builder.add_node("evaluate_documents", evaluate_documents)

    builder.set_entry_point("rag_search")
    builder.add_edge("rag_search", "evaluate_documents")
    builder.add_edge("web_search", "evaluate_documents")

    def route_after_evaluation(state: AgentState) -> str:
        last_message = state["messages"][-1]
        decision = last_message.content
        print(f"[route] Retrieval Team received '{decision}' signal.")

        if decision == "retry_rag":
            return "rag_search"
        if decision in {"fallback_to_web", "retry_web"}:
            return "web_search"
        return END

    builder.add_conditional_edges(
        "evaluate_documents",
        route_after_evaluation,
        {
            "rag_search": "rag_search",
            "web_search": "web_search",
            END: END,
        },
    )

    return builder.compile()
