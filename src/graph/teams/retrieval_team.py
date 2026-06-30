from langgraph.graph import END, StateGraph

from src.agents.retrieval.evaluator import evaluate_documents
from src.agents.retrieval.rag_worker import rag_search
from src.agents.retrieval.web_worker import web_search
from src.schema.constants import NodeName, WorkflowSignal
from src.schema.state import AgentState


def create_retrieval_team_graph():
    builder = StateGraph(AgentState)

    builder.add_node(NodeName.RAG_SEARCH, rag_search)
    builder.add_node(NodeName.WEB_SEARCH, web_search)
    builder.add_node("evaluate_documents", evaluate_documents)

    builder.set_entry_point(NodeName.RAG_SEARCH)
    builder.add_edge(NodeName.RAG_SEARCH, "evaluate_documents")
    builder.add_edge(NodeName.WEB_SEARCH, "evaluate_documents")

    def route_after_evaluation(state: AgentState) -> str:
        last_message = state["messages"][-1]
        decision = last_message.content
        print(f"[route] Retrieval Team received '{decision}' signal.")

        if decision == WorkflowSignal.RETRY_RAG:
            return NodeName.RAG_SEARCH
        if decision in {WorkflowSignal.FALLBACK_TO_WEB, WorkflowSignal.RETRY_WEB}:
            return NodeName.WEB_SEARCH
        return END

    builder.add_conditional_edges(
        "evaluate_documents",
        route_after_evaluation,
        {
            NodeName.RAG_SEARCH: NodeName.RAG_SEARCH,
            NodeName.WEB_SEARCH: NodeName.WEB_SEARCH,
            END: END,
        },
    )

    return builder.compile()
