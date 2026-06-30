from langchain_core.messages import ToolMessage

from src.schema.state import AgentState


def get_query_from_history(state: AgentState) -> str:
    brq = state.get("best_rag_query")
    if brq:
        return brq
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "team1_evaluator":
            return msg.additional_kwargs.get("best_rag_query", "")
    return ""


def get_refined_question_from_history(state: AgentState) -> str:
    q = state.get("q_en_transformed")
    if q:
        return q
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "team1_evaluator":
            return msg.additional_kwargs.get("q_en_transformed", "")
    return ""
