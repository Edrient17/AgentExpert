import uuid
from typing import Any, Dict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

import config
from src.agents.answer.context import get_answer_context
from src.prompts.loader import load_prompt_template
from src.schema.constants import NodeName, is_retry_signal
from src.schema.state import AgentState
from src.tools.common import format_docs
from src.tools.table_image import create_table_image


def _build_feedback_instructions(state: AgentState) -> str:
    feedback = []
    manager_feedback = state.get("manager_feedback")
    last_message = state["messages"][-1]

    if manager_feedback:
        print(f"[feedback] Manager feedback received (Answer Team): {manager_feedback}")
        feedback.append(
            f"""
**IMPORTANT REVISION INSTRUCTION FROM MANAGER:**
Your previous answer was not satisfactory. You MUST revise your answer based on the following feedback:
"{manager_feedback}"
"""
        )

    if (
        isinstance(last_message, ToolMessage)
        and last_message.name == NodeName.ANSWER_EVALUATOR
        and is_retry_signal(last_message.content)
    ):
        internal_feedback = last_message.content.replace("retry:", "").strip()
        if internal_feedback:
            print(f"[feedback] Internal team feedback received (Answer Team): {internal_feedback}")
            feedback.append(
                f"""
**IMPORTANT INTERNAL FEEDBACK FOR REVISION:**
Your previous answer failed the internal quality check. You MUST revise your answer based on the following issue:
"{internal_feedback}"
"""
            )

    return "\n".join(feedback)


def generate_answer(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Answer Generator (answer generation) ---")

    context = get_answer_context(state)
    question = context["q_en_transformed"]
    output_format = context["output_format"]
    docs = context["docs"]
    out_type, answer_language = output_format[0], output_format[1]

    if docs:
        print("... generating answer from retrieved documents")
        prompt = load_prompt_template("answer/generator_with_docs.yaml")
        invoke_params = {
            "user_input": context["user_input"],
            "q_en_transformed": question,
            "out_type": out_type,
            "answer_language": answer_language,
            "rag_context": format_docs(context.get("rag_docs", []) or []),
            "web_context": format_docs(context.get("web_docs", []) or []),
        }
    else:
        print("... generating answer from the LLM's own knowledge")
        prompt = load_prompt_template("answer/generator_general.yaml")
        invoke_params = {
            "user_input": context["user_input"],
            "q_en_transformed": question,
            "out_type": out_type,
            "answer_language": answer_language,
        }

    llm = ChatOpenAI(model=config.LLM_MODEL_TEAM3_GEN, temperature=config.TEAM3_TEMPERATURE)
    chain = prompt.partial(feedback_instructions=_build_feedback_instructions(state)) | llm

    try:
        result = chain.invoke(invoke_params)
        final_content = result.content.strip()

        if out_type == "table" and final_content.startswith("|"):
            try:
                image_path = create_table_image.func(markdown_string=final_content)
                if not image_path.startswith("Error"):
                    final_content += f"\n\n---\n\n**[View generated table image]({image_path})**"
                else:
                    final_content += f"\n\n(Note: table image generation failed - {image_path})"
            except Exception as tool_e:
                print(f"[warn] Direct Table Image Tool call failed: {tool_e}")
                final_content += "\n\n(Note: an error occurred while generating the table image.)"

        return {"manager_feedback": None, "messages": [AIMessage(content=final_content)]}
    except Exception as e:
        print(f"[error] Answer Generator error: {e}")
        return {
            "messages": [
                ToolMessage(
                    content=f"fail: Team3 Worker error - {e}",
                    name=NodeName.ANSWER_WORKER,
                    tool_call_id=str(uuid.uuid4()),
                )
            ]
        }
