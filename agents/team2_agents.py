# agents/team2_agents.py

import json
from typing import List, Dict, Any

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

import config
from state import GlobalState
from utility_tools import vector_store_rag_search, serpapi_web_search, format_docs

# --- Node 1: RAG 검색 (변경 없음) ---

def rag_search(state: GlobalState) -> Dict[str, Any]:
    """
    'vector_store_rag_search' 도구를 호출하여 RAG 검색을 수행합니다.
    """
    print("--- AGENT: Team 2 (RAG 검색) 실행 ---")
    rag_query = state.get("rag_query", "")
    try:
        rag_docs = vector_store_rag_search.invoke(rag_query)
        # 검색 결과가 없어도 실패로 처리하지 않고, 평가 노드에서 판단하도록 그대로 반환
        return {"rag_docs": rag_docs}
    except Exception as e:
        print(f"❌ Team 2 (RAG 검색) 도구 실행 오류: {e}")
        return {"rag_docs": [], "error_message": "RAG 검색 도구 실행에 실패했습니다."}

# --- Node 2: 웹 검색 (변경 없음) ---

def web_search(state: GlobalState) -> Dict[str, Any]:
    """
    'serpapi_web_search' 도구를 호출하여 웹 검색을 수행합니다.
    """
    print("--- AGENT: Team 2 (웹 검색) 실행 ---")
    rag_query = state.get("rag_query", "")
    try:
        web_docs = serpapi_web_search.invoke(rag_query)
        return {"web_docs": web_docs}
    except Exception as e:
        print(f"❌ Team 2 (웹 검색) 도구 실행 오류: {e}")
        return {"web_docs": [], "error_message": "웹 검색 도구 실행에 실패했습니다."}

# --- Node 3: 문서 평가 (Evaluator) ---

class DocEvaluationResult(BaseModel):
    """문서 평가 노드의 LLM 결과 스키마"""
    semantic_relevance: bool
    is_detailed: bool
    error_message: str = ""

def evaluate_documents(state: GlobalState) -> Dict[str, Any]:
    """
    RAG 또는 웹에서 검색된 문서의 품질을 평가합니다.
    """
    print("--- AGENT: Team 2 (문서 평가) 실행 ---")
    q_en_transformed = state.get("q_en_transformed", "")
    rag_query = state.get("rag_query", "")

    current_retries = state.get("team2_retries", 0)
    is_web_search_stage = state.get("web_docs") is not None
    docs_to_evaluate = state.get("web_docs") if is_web_search_stage else state.get("rag_docs", [])
    
    if not docs_to_evaluate:
        return {
            "status": {"team2": "fail"}, 
            "error_message": "Team2: 평가할 문서가 없습니다.",
            "team2_retries": current_retries + 1
        }
    
    docs_preview = format_docs(docs_to_evaluate)

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
- semantic_relevance (bool): Do the docs match the user's intent and constraints?
- is_detailed (bool): Do the docs collectively contain enough specifics to answer the question reliably?
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
            "docs_preview": docs_preview
        })
        result = DocEvaluationResult.model_validate(result_dict)
        passed = result.semantic_relevance and result.is_detailed

        if passed:
            return {"status": {"team2": "pass"}}
        else:
            error_msg = result.error_message or "Team2: 문서 품질 미달"
            
            if not is_web_search_stage and current_retries + 1 >= 2:
                print("... RAG 재시도 소진, 웹 검색으로 전환")
                return {
                    "status": {"team2": "fail"},
                    "error_message": error_msg,
                    "team2_retries": 0,
                    "web_docs": []
                }
            else:
                return {
                    "status": {"team2": "fail"},
                    "error_message": error_msg,
                    "team2_retries": current_retries + 1
                }

    except Exception as e:
        print(f"❌ Team 2 (문서 평가) 오류: {e}")
        return {
            "status": {"team2": "fail"},
            "error_message": f"Team2 Evaluator: 오류 발생 - {e}",
            "team2_retries": current_retries + 1
        }