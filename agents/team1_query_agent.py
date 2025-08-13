# agents/team1_query_agent.py
from typing import List, Dict, Any
from pydantic import BaseModel, Field, ValidationError
import os
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from utils import get_llm

LLM_MODEL = "gpt-4o-mini"

class Team1Result(BaseModel):
    q_validity: bool
    q_en_transformed: str = ""
    rag_queries: List[str] = Field(default_factory=list, min_items=2, max_items=4)
    output_format: List[str] = Field(
        default_factory=lambda: ["qa", "ko"], min_items=2, max_items=2
    )

parser = JsonOutputParser(pydantic_object=Team1Result)

PROMPT = PromptTemplate.from_template("""
You are the first-stage agent in a RAG pipeline.

TASKS
1) q_validity: Decide if the user input is a valid, answerable question (True/False).
   - false if too vague / missing constraints / unsafe.
2) q_en_transformed: Rewrite the question into clear English (preserve domain terms, numbers, units).
3) rag_queries: Generate 2–4 short, diverse, search-friendly English queries (≤15 words each).
   - Mix styles (keyword, semantic paraphrase, entity-focused, time-bounded) when applicable.
   - Do NOT invent facts not implied by the user input. Return 2–4 items only.
4) output_format: ALWAYS return a 2-item array [type, language].
   - type ∈ ["report","table","bulleted","json","qa"]
   - language ∈ ["ko","en"]
   - Defaults apply independently:
       • If type is missing/unclear/invalid → use "qa".
       • If language is missing/unclear/invalid → use "ko".
   - If only one of (type, language) can be inferred, fill the other with its default.
   - Normalize to lowercase. Return exactly two items, no more, no less.

STRICT OUTPUT (JSON ONLY, no prose):
{schema}

USER INPUT:
{user_input}
""").partial(schema=parser.get_format_instructions())

def _build_chain(llm: ChatOpenAI):
    return PROMPT | llm | parser

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
        llm = get_llm(model_name=LLM_MODEL)
        chain = _build_chain(llm)
        
        out = chain.invoke({"user_input": user_input.strip()})
        
        if isinstance(out, Team1Result):
            result = out
        elif isinstance(out, dict):
            result = Team1Result.model_validate(out)
        elif isinstance(out, str):
            result = Team1Result.model_validate(json.loads(out))
        else:
            result = Team1Result.model_validate(json.loads(str(out)))

        rag_queries = [q.strip() for q in result.rag_queries if isinstance(q, str) and q.strip()]

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
        new_state["error_message"] = "Team1: Failed to parse LLM output for question processing."
        return new_state
    except Exception as e:
        print(f"❌ Team 1 Agent error: {e}")
        new_state["error_message"] = f"Team1: An unexpected error occurred in agent_team1: {e}"
        return new_state