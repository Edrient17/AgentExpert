import json
import uuid
from typing import Any, Dict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

import config
from src.agents.answer.context import get_answer_context
from src.prompts.loader import load_prompt_template
from src.schema.outputs import AnswerEvaluationResult
from src.schema.state import AgentState
from src.tools.common import format_docs


def evaluate_answer(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Answer Evaluator (answer evaluation) ---")
    generated_answer_msg = state["messages"][-1]
    if not isinstance(generated_answer_msg, AIMessage):
        return {
            "messages": [
                ToolMessage(
                    content="fail: Could not find an answer to evaluate.",
                    name="team3_evaluator",
                    tool_call_id=str(uuid.uuid4()),
                )
            ]
        }

    current_retries = state.get("team3_retries", 0)
    answer = generated_answer_msg.content
    context = get_answer_context(state)
    question = context["q_en_transformed"]
    output_format = context["output_format"]

    if not all([question, output_format, answer]):
        return {
            "messages": [
                ToolMessage(
                    content="fail: Missing information required for evaluation.",
                    name="team3_evaluator",
                    tool_call_id=str(uuid.uuid4()),
                )
            ]
        }

    parser = JsonOutputParser(p_object=AnswerEvaluationResult)
    prompt = load_prompt_template("answer/evaluator.yaml")
    llm = ChatOpenAI(
        model=config.LLM_MODEL_TEAM3_EVAL,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    chain = prompt.partial(schema=parser.get_format_instructions()) | llm | parser

    try:
        result_dict = chain.invoke(
            {
                "q_en_transformed": question,
                "user_input": context["user_input"],
                "is_simple_query": state.get("is_simple_query", "No"),
                "output_format": json.dumps(output_format, ensure_ascii=False),
                "generated_answer": answer,
                "retrieved_docs": format_docs(context["docs"]),
            }
        )
        result = AnswerEvaluationResult.model_validate(result_dict)
        is_simple = state.get("is_simple_query", "No")

        if is_simple == "Yes":
            passed = result.rules_compliance >= 0.7 and result.question_coverage >= 0.7
        else:
            passed = (
                result.rules_compliance >= 0.7
                and result.question_coverage >= 0.7
                and result.hallucination_score >= 0.7
            )

        if passed:
            return {
                "team3_retries": 0,
                "messages": [
                    ToolMessage(content="pass", name="team3_evaluator", tool_call_id=str(uuid.uuid4()))
                ],
            }

        if current_retries < config.MAX_RETRIES_TEAM3:
            print(f"[retry] Answer Evaluator failed. Requesting retry. ({current_retries + 1}/{config.MAX_RETRIES_TEAM3})")
            err = result.error_message or "Answer quality is insufficient."
            return {
                "team3_retries": current_retries + 1,
                "messages": [
                    ToolMessage(
                        content=f"retry: {err}",
                        name="team3_evaluator",
                        tool_call_id=str(uuid.uuid4()),
                    )
                ],
            }

        print(f"[error] Answer Evaluator final failure (retry limit exceeded: {config.MAX_RETRIES_TEAM3}).")
        return {
            "team3_retries": current_retries + 1,
            "messages": [
                ToolMessage(
                    content="fail: Answer quality is insufficient.",
                    name="team3_evaluator",
                    tool_call_id=str(uuid.uuid4()),
                )
            ],
        }
    except Exception as e:
        print(f"[error] Answer Evaluator error: {e}")
        if current_retries < config.MAX_RETRIES_TEAM3:
            return {
                "team3_retries": current_retries + 1,
                "messages": [
                    ToolMessage(content="retry", name="team3_evaluator", tool_call_id=str(uuid.uuid4()))
                ],
            }
        return {
            "team3_retries": current_retries + 1,
            "messages": [
                ToolMessage(
                    content=f"fail: Team3 Evaluator error - {e}",
                    name="team3_evaluator",
                    tool_call_id=str(uuid.uuid4()),
                )
            ],
        }
