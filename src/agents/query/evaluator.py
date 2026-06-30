import json
import uuid
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

import config
from src.prompts.loader import load_prompt_template
from src.schema.constants import NodeName, WorkflowSignal, fail_signal, retry_signal
from src.schema.outputs import QuestionEvaluationResult
from src.schema.state import AgentState


def evaluate_question(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Query Evaluator (question evaluation) ---")
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.additional_kwargs:
        return {
            "messages": [
                ToolMessage(
                    content=fail_signal("Team1 evaluator could not find the analysis result."),
                    name=NodeName.QUERY_EVALUATOR,
                    tool_call_id=str(uuid.uuid4()),
                )
            ]
        }

    current_retries = state.get("team1_retries", 0)
    processed_data = last_message.additional_kwargs
    user_input = next((msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)), "")

    q_validity = processed_data.get("q_validity", False)
    q_en_transformed = processed_data.get("q_en_transformed", "")
    rag_queries = processed_data.get("rag_queries", [])
    output_format = processed_data.get("output_format", ["qa", "ko"])

    if not q_validity or not all([user_input, q_en_transformed, rag_queries]):
        return {
            "messages": [
                ToolMessage(
                    content=fail_signal("Missing information required for evaluation."),
                    name=NodeName.QUERY_EVALUATOR,
                    tool_call_id=str(uuid.uuid4()),
                )
            ]
        }

    parser = JsonOutputParser(p_object=QuestionEvaluationResult)
    prompt = load_prompt_template("query/question_evaluator.yaml")
    llm = ChatOpenAI(
        model=config.LLM_MODEL_TEAM1,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    chain = prompt.partial(schema=parser.get_format_instructions()) | llm | parser

    try:
        result_dict = chain.invoke(
            {
                "user_input": user_input,
                "q_en_transformed": q_en_transformed,
                "output_format": json.dumps(output_format, ensure_ascii=False),
                "default_format": json.dumps(["qa", "ko"], ensure_ascii=False),
                "rag_queries_json": json.dumps(rag_queries, ensure_ascii=False),
            }
        )
        result = QuestionEvaluationResult.model_validate(result_dict)
        if len(result.rag_query_scores) != len(rag_queries):
            raise ValueError("The score list length does not match the query list length.")

        passed = (result.semantic_alignment >= 0.8) and result.format_compliance
        if passed:
            best_idx = max(range(len(result.rag_query_scores)), key=lambda i: result.rag_query_scores[i])
            best_query = rag_queries[best_idx]
            return {
                "best_rag_query": best_query,
                "q_en_transformed": q_en_transformed,
                "team1_retries": 0,
                "messages": [
                    ToolMessage(
                        content=WorkflowSignal.PASS,
                        name=NodeName.QUERY_EVALUATOR,
                        tool_call_id=str(uuid.uuid4()),
                        additional_kwargs={
                            "q_en_transformed": q_en_transformed,
                            "output_format": output_format,
                            "best_rag_query": best_query,
                        },
                    )
                ],
            }

        err = result.error_message or "Team1: Evaluation criteria not met."
        if current_retries < config.MAX_RETRIES_TEAM1:
            print(f"[retry] Query Evaluator failed. Requesting retry. ({current_retries + 1}/{config.MAX_RETRIES_TEAM1})")
            return {
                "team1_retries": current_retries + 1,
                "messages": [
                    ToolMessage(
                        content=retry_signal(err),
                        name=NodeName.QUERY_EVALUATOR,
                        tool_call_id=str(uuid.uuid4()),
                    )
                ],
            }
        print(f"[error] Query Evaluator final failure (retry limit exceeded: {config.MAX_RETRIES_TEAM1}).")
        return {
            "team1_retries": current_retries + 1,
            "messages": [
                ToolMessage(
                    content=fail_signal(err),
                    name=NodeName.QUERY_EVALUATOR,
                    tool_call_id=str(uuid.uuid4()),
                )
            ],
        }
    except Exception as e:
        print(f"[error] Query Evaluator error: {e}")
        if current_retries < config.MAX_RETRIES_TEAM1:
            return {
                "team1_retries": current_retries + 1,
                "messages": [
                    ToolMessage(content=WorkflowSignal.RETRY, name=NodeName.QUERY_EVALUATOR, tool_call_id=str(uuid.uuid4()))
                ],
            }
        return {
            "team1_retries": current_retries + 1,
            "messages": [
                ToolMessage(
                    content=fail_signal(f"Team1 Evaluator error - {e}"),
                    name=NodeName.QUERY_EVALUATOR,
                    tool_call_id=str(uuid.uuid4()),
                )
            ],
        }
