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

THRESHOLD = 0.7  # 통과 임계치(기존 기준)

# --- 단일 문서 평가 스키마 ---
class DocEvaluationResult(BaseModel):
    semantic_relevance: float = Field(ge=0.0, le=1.0, description="문서가 질문 의도와 제약에 얼마나 관련이 있는지 [0,1]")
    is_detailed: float = Field(ge=0.0, le=1.0, description="문서가 충분히 구체적이고 세부적인지를 나타내는 점수 [0,1]")
    error_message: str = ""

def _get_query_from_history(state: AgentState) -> str:
    for msg in reversed(state['messages']):
        if isinstance(msg, ToolMessage) and msg.name == "team1_evaluator":
            return msg.additional_kwargs.get("best_rag_query", "")
    return ""

def _get_refined_question_from_history(state: AgentState) -> str:
    for msg in reversed(state['messages']):
        if isinstance(msg, ToolMessage) and msg.name == "team1_evaluator":
            return msg.additional_kwargs.get("q_en_transformed", "")
    return ""

# --- Node 1: RAG 검색(10건 확보) ---
def rag_search(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Team 2 (RAG 검색) 실행 ---")
    rag_query = _get_query_from_history(state)
    if not rag_query:
        return {"messages": [ToolMessage(content="fail: RAG 쿼리를 찾을 수 없습니다.", name="rag_search", tool_call_id=str(uuid.uuid4()))]}

    try:
        rag_docs = vector_store_rag_search.func(rag_query, top_k=5, rerank_k=5)  # 5건 평가를 위해 조정
        return {
            "messages": [
                ToolMessage(
                    content=format_docs(rag_docs),
                    name="rag_search_result",
                    tool_call_id=str(uuid.uuid4()),
                    additional_kwargs={"source_docs": rag_docs}
                )
            ],
            # Team2 사이클 시작: 누적 버킷 초기화
            "rag_docs": [],
            "web_docs": [],
        }
    except Exception as e:
        print(f"❌ Team 2 (RAG 검색) 도구 실행 오류: {e}")
        return {"messages": [ToolMessage(content=f"fail: RAG 검색 오류 - {e}", name="rag_search", tool_call_id=str(uuid.uuid4()))]}

# --- Node 2: 웹 검색(3건 단위) ---
def web_search(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Team 2 (웹 검색) 실행 ---")
    rag_query = _get_query_from_history(state)
    try:
        web_docs = deep_research_web_search.func(rag_query, max_results=3)
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

    # 누적 버킷 로드
    rag_acc = list(state.get("rag_docs", []))
    web_acc = list(state.get("web_docs", []))

    # 평가할 대상이 없으면 소스별 기본 분기
    if not docs_to_evaluate:
        decision = "fallback_to_web" if source == "rag" else "retry_web"
        return {
            "messages": [ToolMessage(content=decision, name="team2_evaluator", tool_call_id=str(uuid.uuid4()))],
            "rag_docs": rag_acc,
            "web_docs": web_acc,
        }

    q_en_transformed = _get_refined_question_from_history(state)
    rag_query = _get_query_from_history(state)

    # 단일 문서 평가 체인
    parser = JsonOutputParser(p_object=DocEvaluationResult)
    single_doc_prompt = PromptTemplate.from_template("""
You are the Team2 Supervisor evaluator. Given the question summary and retrieved document,
decide whether the document is good enough to support answering the question.

[Question Summary]
{q_en_transformed}

[RAG Query]
{rag_query}

[Document]
{doc_text}

Return JSON ONLY with the following fields:
- semantic_relevance (float in [0,1]): Do the docs match the user's intent and constraints?
- is_detailed (float in [0,1]): Do the docs collectively contain enough specifics to answer the question reliably?
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
            is_pass = (r.semantic_relevance >= THRESHOLD) and (r.is_detailed >= THRESHOLD)
            if is_pass:
                accepted.append(doc)
            else:
                rejected.append({"reason": r.error_message, "snippet": preview[:300]})
        except Exception as e:
            rejected.append({"reason": f"LLM 오류: {e}", "snippet": (getattr(doc, "page_content", "") or "")[:300]})

    # 소스별 누적
    if accepted:
        if source == "rag":
            rag_acc += accepted
        else:
            web_acc += accepted

    total = len(rag_acc) + len(web_acc)
    print(f"📊 평가 결과: RAG 누적 {len(rag_acc)} / WEB 누적 {len(web_acc)} (합계 {total}, 목표 ≥ 3)")

    if total >= 3:
        # 통과: Team3로 진행
        combined = rag_acc + web_acc  # rag 우선 순서 유지
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
                        # Team3 호환성: 둘 다 전달 + 합본도 함께
                        "rag_docs": rag_acc,
                        "web_docs": web_acc,
                        "retrieved_docs": combined,
                    }
                )
            ],
            "rag_docs": rag_acc,
            "web_docs": web_acc,
        }
    else:
        # 부족: RAG 이후면 웹으로, 웹 이후면 웹 재시도
        decision = "fallback_to_web" if source == "rag" else "retry_web"
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
                    }
                )
            ],
            "rag_docs": rag_acc,
            "web_docs": web_acc,
        }
