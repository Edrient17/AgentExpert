# agents/supervisor_team2.py
import os
import json
from typing import List, Union
from pydantic import BaseModel, ValidationError

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from utils import get_llm, format_docs_for_prompt
from agents.team2_rag_agent import agent_team2_rag_search
from agents.team2_web_agent import agent_team2_web_search

# ===== Eval schema =====
class Team2DocEvalResult(BaseModel):
    semantic_relevance: bool
    is_detailed: bool
    error_message: str = ""

eval_parser = JsonOutputParser(pydantic_object=Team2DocEvalResult)

# ===== Evaluation prompt =====
EVAL_PROMPT = PromptTemplate.from_template("""
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
""").partial(schema=eval_parser.get_format_instructions())

def _eval_docs(llm: ChatOpenAI, q_en: str, query: str, docs: List[Union[str, object]]) -> Team2DocEvalResult:
    """LLM을 이용해 검색된 문서의 품질을 평가."""
    docs_preview = format_docs_for_prompt(docs) if docs else "[NO CONTENT]"
    chain = EVAL_PROMPT | llm | eval_parser
    out = chain.invoke({
        "q_en_transformed": q_en or "",
        "rag_query": query or "",
        "docs_preview": docs_preview,
    })

    if isinstance(out, Team2DocEvalResult):
        return out
    if isinstance(out, dict):
        return Team2DocEvalResult.model_validate(out)
    if isinstance(out, str):
        return Team2DocEvalResult.model_validate(json.loads(out))
    return Team2DocEvalResult.model_validate(json.loads(str(out)))

# ===== Supervisor =====
def supervisor_team2(state: dict) -> dict:
    print("🔍 Team 2 Supervisor 시작")
    state.setdefault("status", {})
    state.setdefault("error_message", "")

    q_summary = state.get("q_en_transformed", "") or ""
    query = state.get("rag_query", "") or ""

    if not query.strip():
        state["status"]["team2"] = "fail"
        state["error_message"] = "Team2: rag_query is empty."
        return state

    max_attempts = 2
    llm = get_llm()

    # === 1) RAG: up to 2 attempts ===
    for attempt in range(1, max_attempts + 1):
        print(f"🔎 RAG 문서 검색 ({attempt}/{max_attempts})")
        rag_state = RunnableLambda(agent_team2_rag_search).invoke(state)
        if isinstance(rag_state, dict):
            state.update(rag_state)

        docs = state.get("rag_docs", []) or []

        if not docs:
            print("... RAG 결과 없음. 웹 검색으로 전환합니다.")
            state["error_message"] = "Team2: RAG results are empty. Falling back to web search."
            break

        try:
            result = _eval_docs(llm, q_summary, query, docs)
        except (ValidationError, OutputParserException) as e:
            print(f"❌ RAG 평가 파싱 오류: {e}")
            state["error_message"] = "Team2: RAG evaluation result parsing failed"
            continue
        except Exception as e:
            print(f"❌ RAG 평가 예외: {e}")
            state["error_message"] = "Team2: Unknown error occurred during RAG evaluation"
            continue

        passed = bool(result.semantic_relevance) and bool(result.is_detailed)
        if passed:
            print("✅ Team 2 통과 (RAG)")
            state["status"]["team2"] = "pass"
            state["error_message"] = ""
            return state
        else:
            print(f"🔁 RAG 평가 미통과 (semantic_relevance={result.semantic_relevance}, is_detailed={result.is_detailed})")
            state["error_message"] = (result.error_message or "Team2: RAG document lacks relevance/detail.").strip()

    # === 2) WEB: up to 2 attempts ===
    print("🌐 WEB 문서 검색 시작")
    for attempt in range(1, max_attempts + 1):
        print(f"-> WEB 문서 검색 ({attempt}/{max_attempts})")
        web_state = RunnableLambda(agent_team2_web_search).invoke(state)
        if isinstance(web_state, dict):
            state.update(web_state)

        docs = state.get("web_docs", []) or []
        if not docs:
            state["error_message"] = "Team2: WEB search results are empty."
            if attempt < max_attempts:
                continue
            else:
                break

        try:
            result = _eval_docs(llm, q_summary, query, docs)
        except (ValidationError, OutputParserException) as e:
            print(f"❌ WEB 평가 파싱 오류: {e}")
            state["error_message"] = "Team2: WEB evaluation result parsing failed"
            continue
        except Exception as e:
            print(f"❌ WEB 평가 예외: {e}")
            state["error_message"] = "Team2: Unknown error occurred during WEB evaluation"
            continue

        passed = bool(result.semantic_relevance) and bool(result.is_detailed)
        if passed:
            print("✅ Team 2 통과 (WEB)")
            state["status"]["team2"] = "pass"
            state["error_message"] = ""
            return state
        else:
            print(f"🔁 WEB 평가 미통과 (semantic_relevance={result.semantic_relevance}, is_detailed={result.is_detailed})")
            state["error_message"] = (result.error_message or "Team2: WEB document lacks relevance/detail.").strip()

    # === All failed ===
    print("❌ Team 2 최종 실패")
    state["status"]["team2"] = "fail"
    if not state.get("error_message") or "RAG" in state["error_message"]:
         state["error_message"] = "Team2: No proper document found after RAG and WEB search."
    return state