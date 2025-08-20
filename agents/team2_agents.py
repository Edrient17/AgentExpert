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
from utility_tools import vector_store_rag_search, serpapi_web_search, format_docs

# --- Pydantic 스키마 ---
class DocEvaluationResult(BaseModel):
    semantic_relevance: float = Field(ge=0.0, le=1.0, description="문서가 질문 의도와 제약에 얼마나 관련이 있는지 [0,1]")
    is_detailed: float = Field(ge=0.0, le=1.0, description="문서가 충분히 구체적이고 세부적인지를 나타내는 점수 [0,1]")
    error_message: str = ""

def _get_query_from_history(state: AgentState) -> str:
    """메시지 히스토리에서 team1_evaluator가 전달한 best_rag_query를 찾습니다."""
    for msg in reversed(state['messages']):
        if isinstance(msg, ToolMessage) and msg.name == "team1_evaluator":
            return msg.additional_kwargs.get("best_rag_query", "")
    return ""

# --- Node 1 & 2: RAG 및 웹 검색 ---
def rag_search(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Team 2 (RAG 검색) 실행 ---")

    rag_query = _get_query_from_history(state)
    if not rag_query:
        return {"messages": [ToolMessage(content="fail: RAG 쿼리를 찾을 수 없습니다.", name="rag_search", tool_call_id=str(uuid.uuid4()))]}
        
    try:
        rag_docs = vector_store_rag_search.func(rag_query)
        return {
            "messages": [
                ToolMessage(
                    content=format_docs(rag_docs),
                    name="rag_search_result",
                    tool_call_id=str(uuid.uuid4()), 
                    additional_kwargs={"source_docs": rag_docs}
                )
            ]
        }
    except Exception as e:
        print(f"❌ Team 2 (RAG 검색) 도구 실행 오류: {e}")
        return {"messages": [ToolMessage(content=f"fail: RAG 검색 오류 - {e}", name="rag_search", tool_call_id=str(uuid.uuid4()))]}

def web_search(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Team 2 (웹 검색) 실행 ---")
    rag_query = _get_query_from_history(state)
    try:
        web_docs = serpapi_web_search.func(rag_query)
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

# --- Node 3: 문서 평가 (Evaluator) ---
def evaluate_documents(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Team 2 (문서 평가) 실행 ---")
    last_message = state['messages'][-1]
    
    docs_to_evaluate = last_message.additional_kwargs.get("source_docs", [])
    search_source = "web_search" if last_message.name == "web_search_result" else "rag_search"

    current_retries = state.get("team2_retries", 0)

    if not docs_to_evaluate:
        passed = False
        error_msg = "평가할 문서가 없습니다."
    else:
        q_en_transformed, rag_query = "", ""
        for msg in reversed(state['messages']):
            if isinstance(msg, ToolMessage) and msg.name == "team1_evaluator":
                q_en_transformed = msg.additional_kwargs.get("q_en_transformed", "")
                rag_query = msg.additional_kwargs.get("best_rag_query", "")
                break

        parser = JsonOutputParser(p_object=DocEvaluationResult)
        prompt = PromptTemplate.from_template("""
You are the Team2 Supervisor evaluator. Given the question summary and retrieved docs,
decide whether the docs are good enough to support answering the question.

[Question Summary]
{q_en_transformed}

[RAG Query]
{rag_query}

[Retrieved Docs Preview]  # concatenated & possibly truncated
{docs_preview}

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
        chain = prompt | llm | parser

        try:
            result_dict = chain.invoke({
                "q_en_transformed": q_en_transformed,
                "rag_query": rag_query,
                "docs_preview": format_docs(docs_to_evaluate)
            })
            result = DocEvaluationResult.model_validate(result_dict)
            passed = (result.semantic_relevance >= 0.7) and (result.is_detailed >= 0.7)
            error_msg = result.error_message or "Team2: 문서 품질 미달"
        except Exception as e:
            print(f"❌ Team 2 (문서 평가) LLM 오류: {e}")
            passed = False
            error_msg = f"Team2 Evaluator LLM 오류 - {e}"

    if passed:
        return {
            "messages": [
                ToolMessage(
                    content="pass", 
                    name="team2_evaluator",
                    tool_call_id=str(uuid.uuid4()),
                    additional_kwargs={
                        "source": search_source,
                        "retrieved_docs": docs_to_evaluate
                    }
                )
            ],
            "team2_retries": current_retries + 1
        }
    else: # 평가 실패 시
        if current_retries == 0:
            print(f"🔁 RAG 평가 실패. 재시도를 요청합니다. ({current_retries + 1}/{config.MAX_RETRIES_TEAM2})")
            return {
                "messages": [ToolMessage(content="retry_rag", name="team2_evaluator", tool_call_id=str(uuid.uuid4()))],
                "team2_retries": current_retries + 1
            }
        elif current_retries == 1:
            print(f"🔁 RAG 최종 실패. 웹 검색으로 대체를 요청합니다.")
            return {
                "messages": [ToolMessage(content="fallback_to_web", name="team2_evaluator", tool_call_id=str(uuid.uuid4()))],
                "team2_retries": current_retries + 1
            }
        elif current_retries < config.MAX_RETRIES_TEAM2:
            print(f"🔁 WEB 평가 실패. 재시도를 요청합니다. ({current_retries + 1}/{config.MAX_RETRIES_TEAM2})")
            return {
                "messages": [ToolMessage(content="retry_web", name="team2_evaluator", tool_call_id=str(uuid.uuid4()))],
                "team2_retries": current_retries + 1
            }
        else:
            print(f"❌ WEB 최종 실패.")
            return {
                "messages": [ToolMessage(content=f"fail: {error_msg}", name="team2_evaluator", tool_call_id=str(uuid.uuid4()))],
                "team2_retries": current_retries + 1
            }