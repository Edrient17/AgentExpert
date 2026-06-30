import uuid
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

import config
from src.prompts.loader import load_prompt_template
from src.schema.outputs import QuestionProcessingResult
from src.schema.state import AgentState
from src.tools.query_classifier import classify_simple_query


def _build_feedback_instructions(state: AgentState) -> str:
    feedback = []
    manager_feedback = state.get("manager_feedback")
    last_message = state["messages"][-1]

    if manager_feedback:
        print(f"[feedback] Manager feedback received: {manager_feedback}")
        feedback.append(
            f"""
**IMPORTANT REVISION INSTRUCTION FROM MANAGER:**
The previous attempt was not good enough. You MUST incorporate the following feedback into your new result:
"{manager_feedback}"
"""
        )

    if (
        isinstance(last_message, ToolMessage)
        and last_message.name == "team1_evaluator"
        and last_message.content.startswith("retry")
    ):
        internal_feedback = last_message.content.replace("retry:", "").strip()
        if internal_feedback:
            print(f"[feedback] Internal team feedback received: {internal_feedback}")
            feedback.append(
                f"""
**IMPORTANT INTERNAL FEEDBACK FOR REVISION:**
Your previous attempt failed the internal quality check. You MUST address the following issue:
"{internal_feedback}"
"""
            )

    return "\n".join(feedback)


def process_question(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Query Worker (question processing) ---")

    user_input = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), "")
    if not user_input.strip():
        return {
            "messages": [
                ToolMessage(
                    content="fail: No question was provided.",
                    name="team1_worker",
                    tool_call_id=str(uuid.uuid4()),
                )
            ]
        }

    try:
        check_simple = classify_simple_query.func(user_input)
        print(f"[tool] classify_simple_query(LLM) -> {check_simple}")
    except Exception as e:
        print(f"[warn] classify_simple_query failed: {e}")
        check_simple = "No"

    parser = JsonOutputParser(p_object=QuestionProcessingResult)
    prompt = load_prompt_template("query/question_processor.yaml")
    llm = ChatOpenAI(
        model=config.LLM_MODEL_TEAM1,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    chain = prompt.partial(
        feedback_instructions=_build_feedback_instructions(state),
        schema=parser.get_format_instructions(),
    ) | llm | parser

    try:
        result_dict = chain.invoke({"user_input": user_input})
        if not result_dict.get("rag_queries"):
            raise ValueError("rag_queries is empty.")
        return {
            "user_input": user_input,
            "q_en_transformed": result_dict.get("q_en_transformed", ""),
            "is_simple_query": check_simple,
            "manager_feedback": None,
            "messages": [
                AIMessage(
                    content="The user question was analyzed successfully.",
                    additional_kwargs=result_dict,
                )
            ],
        }
    except Exception as e:
        print(f"[error] Query Worker error: {e}")
        return {
            "messages": [
                ToolMessage(
                    content=f"fail: Team1 Worker error - {e}",
                    name="team1_worker",
                    tool_call_id=str(uuid.uuid4()),
                )
            ]
        }
