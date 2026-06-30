from langchain_core.messages import HumanMessage, ToolMessage

from src.schema.state import AgentState


def get_answer_context(state: AgentState) -> dict:
    context = {
        "user_input": state.get("user_input", ""),
        "q_en_transformed": "",
        "output_format": ["qa", "ko"],
        "rag_docs": [],
        "web_docs": [],
        "docs": [],
    }

    if not context["user_input"]:
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                context["user_input"] = msg.content
                break

    rag_from_state = state.get("rag_docs", [])
    web_from_state = state.get("web_docs", [])
    if rag_from_state or web_from_state:
        context["rag_docs"] = rag_from_state
        context["web_docs"] = web_from_state
        context["docs"] = rag_from_state + web_from_state

    if not context["docs"]:
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage) and msg.name == "team2_evaluator" and msg.content == "pass":
                rag_docs = msg.additional_kwargs.get("rag_docs", [])
                web_docs = msg.additional_kwargs.get("web_docs", [])
                if rag_docs or web_docs:
                    context["rag_docs"] = rag_docs
                    context["web_docs"] = web_docs
                    context["docs"] = rag_docs + web_docs
                else:
                    context["docs"] = msg.additional_kwargs.get("retrieved_docs", [])
                break

    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.name == "team1_evaluator":
            context["q_en_transformed"] = msg.additional_kwargs.get("q_en_transformed", "")
            context["output_format"] = msg.additional_kwargs.get("output_format", ["qa", "ko"])
            break

    return context
