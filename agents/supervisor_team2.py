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

from agents.team2_rag_agent import agent_team2_rag_search
from agents.team2_web_agent import agent_team2_web_search

# ===== Eval schema: booleans only =====
class Team2DocEvalResult(BaseModel):
    semantic_relevance: bool
    is_detailed: bool
    error_message: str = ""  # 문제가 있으면 간결 한국어, 없으면 ""

eval_parser = JsonOutputParser(pydantic_object=Team2DocEvalResult)

# ===== Evaluation prompt (EN) =====
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

# ===== Lazy LLM =====
def _get_llm() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ===== Docs util =====
def _combine_docs(docs: List[Union[str, object]], max_chars: int = 6000) -> str:
    """Accepts langchain Document or str. Concatenate with truncation."""
    contents: List[str] = []
    for d in docs:
        try:
            text = getattr(d, "page_content", None)
            if text is None and isinstance(d, str):
                text = d
            if text:
                text = str(text).strip()
                if text:
                    contents.append(text)
        except Exception:
            continue
    joined = "\n\n---\n\n".join(contents)
    if len(joined) > max_chars:
        return joined[:max_chars] + "\n\n...[truncated]..."
    return joined

def _eval_docs(q_en: str, query: str, docs: List[Union[str, object]]) -> Team2DocEvalResult:
    docs_preview = _combine_docs(docs) if docs else "[NO CONTENT]"
    llm = _get_llm()
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
        state["error_message"] = "Team2: rag_query가 비어 있어 검색을 수행할 수 없습니다."
        return state

    max_attempts = 2

    # === 1) RAG: up to 2 attempts ===
    for attempt in range(1, max_attempts + 1):
        print(f"🔎 RAG 문서 검색 ({attempt}/{max_attempts})")
        rag_state = RunnableLambda(agent_team2_rag_search).invoke(state)
        if isinstance(rag_state, dict):
            state.update(rag_state)  # merge safely

        docs = state.get("rag_docs", []) or []
        if not docs:
            state["error_message"] = "Team2: RAG 결과가 비어 있습니다."

        try:
            result = _eval_docs(q_summary, query, docs)
        except (ValidationError, OutputParserException) as e:
            print(f"❌ RAG 평가 파싱 오류: {e}")
            state["error_message"] = "Team2: RAG 평가 결과 파싱 실패"
            continue
        except Exception as e:
            print(f"❌ RAG 평가 예외: {e}")
            state["error_message"] = "Team2: RAG 평가 중 알 수 없는 오류"
            continue

        passed = bool(result.semantic_relevance) and bool(result.is_detailed)
        if passed:
            print("✅ Team 2 통과 (RAG)")
            state["status"]["team2"] = "pass"
            state["error_message"] = ""
            return state
        else:
            print("🔁 RAG 평가 미통과 (semantic_relevance="
                  f"{result.semantic_relevance}, is_detailed={result.is_detailed})")
            state["error_message"] = (result.error_message or
                                      "Team2: RAG 문서가 관련성/세부성이 부족합니다.").strip()

    # === 2) WEB: up to 2 attempts ===
    for attempt in range(1, max_attempts + 1):
        print(f"🌐 WEB 문서 검색 ({attempt}/{max_attempts})")
        web_state = RunnableLambda(agent_team2_web_search).invoke(state)
        if isinstance(web_state, dict):
            state.update(web_state)  # merge safely

        docs = state.get("web_docs", []) or []
        if not docs:
            state["error_message"] = "Team2: 웹 검색 결과가 비어 있습니다."

        try:
            result = _eval_docs(q_summary, query, docs)
        except (ValidationError, OutputParserException) as e:
            print(f"❌ WEB 평가 파싱 오류: {e}")
            state["error_message"] = "Team2: WEB 평가 결과 파싱 실패"
            continue
        except Exception as e:
            print(f"❌ WEB 평가 예외: {e}")
            state["error_message"] = "Team2: WEB 평가 중 알 수 없는 오류"
            continue

        passed = bool(result.semantic_relevance) and bool(result.is_detailed)
        if passed:
            print("✅ Team 2 통과 (WEB)")
            state["status"]["team2"] = "pass"
            state["error_message"] = ""
            return state
        else:
            print("🔁 WEB 평가 미통과 (semantic_relevance="
                  f"{result.semantic_relevance}, is_detailed={result.is_detailed})")
            state["error_message"] = (result.error_message or
                                      "Team2: 웹 문서가 관련성/세부성이 부족합니다.").strip()

    # === All failed ===
    print("❌ Team 2 최종 실패")
    state["status"]["team2"] = "fail"
    if not state.get("error_message"):
        state["error_message"] = "Team2: 적절한 문서를 찾지 못했습니다."
    return state
