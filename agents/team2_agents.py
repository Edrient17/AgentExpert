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

semantic_relevance_THRESHOLD = 0.5
is_detailed_THRESHOLD = 0.5
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
    q_en_transformed = state.get("q_en_transformed", "")
    try:
        web_docs = deep_research_web_search.func(q_en_transformed, max_results=web_search_num)
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
You are a strict Quality Control Supervisor evaluator.
Your job is to carefully assess whether the given document is sufficiently relevant and detailed to help answer the user’s question.
Follow the instructions below without deviation and return ONLY a valid JSON object matching the schema.

[Evaluation Guidelines]
- Use the two scoring rubrics below independently: one for semantic_relevance and one for is_detailed.
- Judge only based on the provided inputs. Do not invent information.
- All contents of the document should be considered, not just part of the document.

[Scoring Guide — semantic_relevance]
Choose EXACTLY one level for how well the document matches the question’s intent and constraints (subject, entities, context).
- 0.00 = None: completely irrelevant or empty; contradicts the question or ignores core entities/constraints.
- 0.25 = Low: superficial keyword overlap; misses main intent or key constraints; noticeable topic drift.
- 0.50 = Partial: addresses the main topic but misses important constraints/context; mixed or uneven relevance.
- 0.75 = Strong: satisfies most intent/constraints with minor mismatches or small gaps.
- 1.00 = Exact: fully aligned with the question’s entities and constraints; no topic drift.

[Scoring Guide — is_detailed]
Choose EXACTLY one level for how specific and sufficient the document is to support a reliable answer.
- 0.00 = None: empty/generic; no actionable specifics.
- 0.25 = Low: few specifics; vague statements; lacks steps, data, or concrete facts.
- 0.50 = Partial: some specifics but missing key details to answer fully; incomplete coverage.
- 0.75 = Strong: solid specifics; covers most needed details with minor gaps.
- 1.00 = Exact: comprehensive and specific (e.g., steps, parameters, examples, citations, or numbers); fully sufficient.

[Error Message]
- If the document is empty, irrelevant, duplicated, off-topic, or too generic/outdated for the question, write a short English note (<= 20 words).
- Otherwise, return "".

[Inputs]
- Question Summary: {q_en_transformed}
- RAG Query: {rag_query}
- Document (excerpted for evaluation): {doc_text}

[Output Instructions]
- Return ONLY a valid JSON object; no commentary, Markdown, code fences, or extra keys.
- Keys must exactly match the schema fields.
- Values for the two scores MUST be one of: 0.00, 0.25, 0.50, 0.75, 1.00.

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