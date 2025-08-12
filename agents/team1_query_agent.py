# agents/team1_query_agent.py
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
import os
import json

LLM_MODEL = "gpt-4o-mini"

# 1) 출력 스키마: rag_queries(2–4개), output_format은 항상 2개짜리 리스트 [type, language]
class Team1Result(BaseModel):
    q_validity: bool
    q_en_transformed: str = ""
    rag_queries: List[str] = Field(default_factory=list, min_items=2, max_items=4)
    output_format: List[str] = Field(
        default_factory=lambda: ["qa", "ko"], min_items=2, max_items=2
    )

parser = JsonOutputParser(pydantic_object=Team1Result)

# 2) 프롬프트
PROMPT = PromptTemplate.from_template("""
You are the first-stage agent in a RAG pipeline.

TASKS
1) q_validity: Decide if the user input is a valid, answerable question (True/False).
   - false if too vague / missing constraints / unsafe.
2) q_en_transformed: Rewrite the question into clear English (preserve domain terms, numbers, units).
3) rag_queries: Generate 2–4 short, diverse, search-friendly English queries (≤15 words each).
   - Mix styles (keyword, semantic paraphrase, entity-focused, time-bounded) when applicable.
   - Do NOT invent facts not implied by the user input. Return 2–4 items only.
4) output_format: ALWAYS output a 2-item array [type, language].
   - type ∈ ["report","table","bulleted","json","qa"]
   - language ∈ ["ko","en"]
   - If unclear, use ["qa","ko"].

OUTPUT (JSON ONLY):
{schema}

USER INPUT:
{user_input}
""").partial(schema=parser.get_format_instructions())

# === LLM 지연 생성 (import 시점 X) ===
def _get_llm() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
    return ChatOpenAI(model=LLM_MODEL, temperature=0)

def _build_chain():
    llm = _get_llm()
    return PROMPT | llm | parser

# 4) 에이전트 함수
def agent_team1_question_processing(state: Dict[str, Any]) -> Dict[str, Any]:
    user_input = state.get("user_input", "")
    new_state = state.copy()

    if not isinstance(user_input, str) or not user_input.strip():
        new_state["q_validity"] = False
        new_state["q_en_transformed"] = ""
        new_state["rag_queries"] = []
        new_state["output_format"] = ["qa", "ko"]
        return new_state

    try:
        chain = _build_chain()
        out = chain.invoke({"user_input": user_input.strip()})
        # ✅ 반환 정규화: dict / BaseModel / str(JSON) 모두 수용
        if isinstance(out, Team1Result):
            result = out
        elif isinstance(out, dict):
            result = Team1Result.model_validate(out)  # pydantic v2
        elif isinstance(out, str):
            result = Team1Result.model_validate(json.loads(out))
        else:
            # 알 수 없는 타입이면 문자열화 -> JSON 시도
            result = Team1Result.model_validate(json.loads(str(out)))

        # rag_queries 정리
        rag_queries = [q.strip() for q in result.rag_queries if isinstance(q, str) and q.strip()]

        # output_format 정규화
        allowed_types = {"report", "table", "bulleted", "json", "qa"}
        allowed_langs = {"ko", "en"}
        t = (result.output_format[0] if len(result.output_format) >= 1 else "qa").strip().lower()
        l = (result.output_format[1] if len(result.output_format) >= 2 else "ko").strip().lower()
        if t not in allowed_types: t = "qa"
        if l not in allowed_langs: l = "ko"

        new_state["q_validity"] = bool(result.q_validity)
        new_state["q_en_transformed"] = result.q_en_transformed.strip()
        new_state["rag_queries"] = rag_queries
        new_state["output_format"] = [t, l]
        return new_state

    except (ValidationError, OutputParserException) as e:
        print(f"❌ Team 1 parsing error: {e}")
        return new_state            # ← 예외 시 바로 반환(초기값 유지)
    except Exception as e:
        print(f"❌ Team 1 Agent error: {e}")
        return new_state            # ← 예외 시 바로 반환(초기값 유지)
