# agents/team2_agents.py

import json
import uuid
from typing import List, Dict, Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

import config
from state import AgentState
from utility_tools import vector_store_rag_search, deep_research_web_search, format_docs

semantic_relevance_THRESHOLD = 0.7
is_detailed_THRESHOLD = 0.8
rag_search_num = 7
web_search_num = 5
total_docs_required = 5

# --- 단일 문서 평가 스키마 ---
class DocEvaluationResult(BaseModel):
    semantic_relevance: float = Field(ge=0.0, le=1.0, description="문서가 질문 의도와 제약에 얼마나 관련이 있는지 [0,1]")
    is_detailed: float = Field(ge=0.0, le=1.0, description="문서가 충분히 구체적이고 세부적인지를 나타내는 점수 [0,1]")
    error_message: str = ""

def _get_query_from_history(state: AgentState) -> str:
    brq = state.get("best_rag_query")
    if brq:
        return brq
    for msg in reversed(state['messages']):
        if isinstance(msg, ToolMessage) and msg.name == "team1_evaluator":
            return msg.additional_kwargs.get("best_rag_query", "")
    return ""

def _get_refined_question_from_history(state: AgentState) -> str:
    q = state.get("q_en_transformed")
    if q:
        return q
    for msg in reversed(state['messages']):
        if isinstance(msg, ToolMessage) and msg.name == "team1_evaluator":
            return msg.additional_kwargs.get("q_en_transformed", "")
    return ""

# --- Node 1: RAG 검색 ---
def rag_search(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Team 2 (RAG 검색) 실행 ---")
    rag_query = _get_query_from_history(state)
    if not rag_query:
        return {"messages": [ToolMessage(content="fail: RAG 쿼리를 찾을 수 없습니다.", name="rag_search", tool_call_id=str(uuid.uuid4()))]}

    try:
        rag_docs = vector_store_rag_search.func(rag_query, top_k=rag_search_num, rerank_k=rag_search_num)
        return {
            "messages": [
                ToolMessage(
                    content=format_docs(rag_docs),
                    name="rag_search_result",
                    tool_call_id=str(uuid.uuid4()),
                    additional_kwargs={"source_docs": rag_docs}
                )
            ],

            "rag_docs": [],
            "web_docs": [],
        }
    except Exception as e:
        print(f"❌ Team 2 (RAG 검색) 도구 실행 오류: {e}")
        return {"messages": [ToolMessage(content=f"fail: RAG 검색 오류 - {e}", name="rag_search", tool_call_id=str(uuid.uuid4()))]}

# --- Node 2: 웹 검색 ---
def web_search(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Team 2 (웹 검색) 실행 ---")
    rag_query = _get_query_from_history(state)
    try:
        web_docs = deep_research_web_search.func(rag_query, max_results=web_search_num)
        return {
            "messages": [
                ToolMessage(
                    content=format_docs(web_docs),
                    name="web_search_result",
                    tool_call_id=str(uuid.uuid4()),
                    additional_kwargs={"source_docs": web_docs}
                )
            ]
        }
    except Exception as e:
        print(f"❌ Team 2 (웹 검색) 도구 실행 오류: {e}")
        return {"messages": [ToolMessage(content=f"fail: 웹 검색 오류 - {e}", name="web_search", tool_call_id=str(uuid.uuid4()))]}

# --- Node 3: 문서 평가(문서별 스코어링 & 소스별 누적) ---
def evaluate_documents(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Team 2 (문서 평가) 실행 ---")

    last_message = state['messages'][-1]
    docs_to_evaluate = last_message.additional_kwargs.get("source_docs", [])
    source = "web" if last_message.name == "web_search_result" else "rag"

    rag_acc = list(state.get("rag_docs", []))
    web_acc = list(state.get("web_docs", []))

    current_retries = state.get("team2_retries", 0)

    if not docs_to_evaluate:
        decision = "fallback_to_web" if source == "rag" else "retry_web"
        next_retries = current_retries + 1
        if next_retries >= config.MAX_RETRIES_TEAM2:
            decision = "fail"
        return {
            "messages": [
                ToolMessage(
                    content=decision,
                    name="team2_evaluator",
                    tool_call_id=str(uuid.uuid4()),
                    additional_kwargs={
                        "source": source,
                        "accepted_rag": len(rag_acc),
                        "accepted_web": len(web_acc),
                        "current_total": len(rag_acc) + len(web_acc),
                        "retries": next_retries,
                        "max_retries": config.MAX_RETRIES_TEAM2,
                        "failed_reason": "no_docs_to_evaluate" if decision == "fail" else ""
                    }
                )
            ],
            "rag_docs": rag_acc,
            "web_docs": web_acc,
            "team2_retries": next_retries,
        }

    q_en_transformed = _get_refined_question_from_history(state)
    rag_query = _get_query_from_history(state)

    parser = JsonOutputParser(p_object=DocEvaluationResult)
    single_doc_prompt = PromptTemplate.from_template("""
You are the strict Quality Control Supervisor evaluator. Given the question summary and document,
decide whether the document is good enough to support answering the question.

[Question Summary]
{q_en_transformed}

[RAG Query]
{rag_query}

[Document]
{doc_text}

Return JSON ONLY with the following fields:
- semantic_relevance (float in [0,1]): Do the docs match the user's intent and constraints?
- is_detailed (float in [0,1]): Do the docs provide enough specific details to comprehensively and reliably answer all parts of the question?
- error_message (str): If anything is wrong (empty/irrelevant/too generic/duplicated), write a short Korean message; else "".

Output schema:
{schema}
""").partial(schema=parser.get_format_instructions())
    llm = ChatOpenAI(
        model=config.LLM_MODEL_TEAM2_EVAL,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    chain = single_doc_prompt | llm | parser

    accepted: List[Any] = []
    rejected: List[Any] = []

    for doc in docs_to_evaluate:
        try:
            preview = (getattr(doc, "page_content", "") or "")[:4000]
            result_dict = chain.invoke({"q_en_transformed": q_en_transformed, "rag_query": rag_query, "doc_text": preview})
            r = DocEvaluationResult.model_validate(result_dict)
            is_pass = (r.semantic_relevance >= semantic_relevance_THRESHOLD) and (r.is_detailed >= is_detailed_THRESHOLD)
            if is_pass:
                accepted.append(doc)
            else:
                rejected.append({"reason": r.error_message, "snippet": preview[:300]})
        except Exception as e:
            rejected.append({"reason": f"LLM 오류: {e}", "snippet": (getattr(doc, "page_content", "") or "")[:300]})

    if accepted:
        if source == "rag":
            rag_acc += accepted
        else:
            web_acc += accepted

    total = len(rag_acc) + len(web_acc)
    print(f"📊 평가 결과: RAG 누적 {len(rag_acc)} / WEB 누적 {len(web_acc)} (합계 {total}, 목표 ≥ {total_docs_required})")

    if total >= total_docs_required:
        combined = rag_acc + web_acc
        return {
            "messages": [
                ToolMessage(
                    content="pass",
                    name="team2_evaluator",
                    tool_call_id=str(uuid.uuid4()),
                    additional_kwargs={
                        "source": source,
                        "accepted_rag": len(rag_acc),
                        "accepted_web": len(web_acc),
                        "rag_docs": rag_acc,
                        "web_docs": web_acc,
                        "retrieved_docs": combined,
                    }
                )
            ],
            "rag_docs": rag_acc,
            "web_docs": web_acc,
            "team2_retries": 0,  # ✅ 리셋
        }
    else:
        decision = "fallback_to_web" if source == "rag" else "retry_web"
        next_retries = current_retries + 1
        if next_retries >= config.MAX_RETRIES_TEAM2:
            decision = "fail"
        return {
            "messages": [
                ToolMessage(
                    content=decision,
                    name="team2_evaluator",
                    tool_call_id=str(uuid.uuid4()),
                    additional_kwargs={
                        "source": source,
                        "accepted_rag": len(rag_acc),
                        "accepted_web": len(web_acc),
                        "current_total": total,
                        "retries": next_retries,
                        "max_retries": config.MAX_RETRIES_TEAM2,
                        "failed_reason": "budget_exhausted" if decision == "fail" else ""
                    }
                )
            ],
            "rag_docs": rag_acc,
            "web_docs": web_acc,
            "team2_retries": next_retries,
        }