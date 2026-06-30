import uuid
from typing import Any, Dict, List

from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

import config
from src.agents.retrieval.context import get_query_from_history, get_refined_question_from_history
from src.prompts.loader import load_prompt_template
from src.schema.constants import NodeName, RetrievalSource, WorkflowSignal
from src.schema.outputs import DocEvaluationResult
from src.schema.state import AgentState


SEMANTIC_RELEVANCE_THRESHOLD = 0.5
IS_DETAILED_THRESHOLD = 0.5
TOTAL_DOCS_REQUIRED = 5


def _build_decision_payload(
    decision: str,
    source: str,
    rag_acc: List[Any],
    web_acc: List[Any],
    retries: int,
    failed_reason: str = "",
) -> Dict[str, Any]:
    return {
        "messages": [
            ToolMessage(
                content=decision,
                name=NodeName.RETRIEVAL_EVALUATOR,
                tool_call_id=str(uuid.uuid4()),
                additional_kwargs={
                    "source": source,
                    "accepted_rag": len(rag_acc),
                    "accepted_web": len(web_acc),
                    "current_total": len(rag_acc) + len(web_acc),
                    "retries": retries,
                    "max_retries": config.MAX_RETRIES_TEAM2,
                    "failed_reason": failed_reason,
                },
            )
        ],
        "rag_docs": rag_acc,
        "web_docs": web_acc,
        "team2_retries": retries,
    }


def evaluate_documents(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Retrieval Evaluator (document evaluation) ---")

    last_message = state["messages"][-1]
    docs_to_evaluate = last_message.additional_kwargs.get("source_docs", [])
    source = RetrievalSource.WEB if last_message.name == NodeName.WEB_SEARCH_RESULT else RetrievalSource.RAG

    rag_acc = list(state.get("rag_docs", []))
    web_acc = list(state.get("web_docs", []))
    current_retries = state.get("team2_retries", 0)

    if not docs_to_evaluate:
        decision = WorkflowSignal.FALLBACK_TO_WEB if source == RetrievalSource.RAG else WorkflowSignal.RETRY_WEB
        next_retries = current_retries + 1
        failed_reason = ""
        if source == RetrievalSource.RAG and not config.ENABLE_WEB_RESEARCH:
            decision = WorkflowSignal.FAIL
            failed_reason = "web_research_disabled"
        if next_retries >= config.MAX_RETRIES_TEAM2:
            decision = WorkflowSignal.FAIL
            failed_reason = "no_docs_to_evaluate"
        return _build_decision_payload(decision, source, rag_acc, web_acc, next_retries, failed_reason)

    q_en_transformed = get_refined_question_from_history(state)
    rag_query = get_query_from_history(state)

    parser = JsonOutputParser(p_object=DocEvaluationResult)
    prompt = load_prompt_template("retrieval/document_evaluator.yaml")
    llm = ChatOpenAI(
        model=config.LLM_MODEL_TEAM2_EVAL,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    chain = prompt.partial(schema=parser.get_format_instructions()) | llm | parser

    accepted: List[Any] = []

    for doc in docs_to_evaluate:
        try:
            preview = (getattr(doc, "page_content", "") or "")[:4000]
            result_dict = chain.invoke(
                {
                    "q_en_transformed": q_en_transformed,
                    "rag_query": rag_query,
                    "doc_text": preview,
                }
            )
            result = DocEvaluationResult.model_validate(result_dict)
            is_pass = (
                result.semantic_relevance >= SEMANTIC_RELEVANCE_THRESHOLD
                and result.is_detailed >= IS_DETAILED_THRESHOLD
            )
            if is_pass:
                accepted.append(doc)
        except Exception as e:
            print(f"[warn] Document evaluation failed: {e}")

    if accepted:
        if source == "rag":
            rag_acc += accepted
        else:
            web_acc += accepted

    total = len(rag_acc) + len(web_acc)
    print(f"[stats] Evaluation result: RAG accepted {len(rag_acc)} / WEB accepted {len(web_acc)} (total {total}, target >= {TOTAL_DOCS_REQUIRED})")

    if total >= TOTAL_DOCS_REQUIRED:
        combined = rag_acc + web_acc
        return {
            "messages": [
                ToolMessage(
                    content=WorkflowSignal.PASS,
                    name=NodeName.RETRIEVAL_EVALUATOR,
                    tool_call_id=str(uuid.uuid4()),
                    additional_kwargs={
                        "source": source,
                        "accepted_rag": len(rag_acc),
                        "accepted_web": len(web_acc),
                        "rag_docs": rag_acc,
                        "web_docs": web_acc,
                        "retrieved_docs": combined,
                    },
                )
            ],
            "rag_docs": rag_acc,
            "web_docs": web_acc,
            "team2_retries": 0,
        }

    decision = WorkflowSignal.FALLBACK_TO_WEB if source == RetrievalSource.RAG else WorkflowSignal.RETRY_WEB
    next_retries = current_retries + 1
    failed_reason = ""
    if source == RetrievalSource.RAG and not config.ENABLE_WEB_RESEARCH:
        decision = WorkflowSignal.FAIL
        failed_reason = "web_research_disabled"
    if next_retries >= config.MAX_RETRIES_TEAM2:
        decision = WorkflowSignal.FAIL
        failed_reason = "budget_exhausted"
    return _build_decision_payload(decision, source, rag_acc, web_acc, next_retries, failed_reason)
