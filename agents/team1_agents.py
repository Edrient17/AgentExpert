# agents/team1_agents.py

import json
from typing import List, Dict, Any

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

import config
from state import GlobalState

# --- Node 1: 질문 처리 (Worker) ---

class QuestionProcessingResult(BaseModel):
    """질문 처리 노드의 결과 스키마"""
    q_validity: bool = Field(description="사용자 질문이 답변 가능한 유효한 질문인지 여부 (True/False)")
    q_en_transformed: str = Field(description="사용자 질문을 명확하게 재구성한 영문 질문")
    rag_queries: List[str] = Field(description="검색에 사용할 2-4개의 다양한 영문 RAG 쿼리 후보 리스트", min_items=2, max_items=4)
    output_format: List[str] = Field(description="요청된 출력 포맷 [type, language]", min_items=2, max_items=2)

def process_question(state: GlobalState) -> Dict[str, Any]:
    """
    사용자의 입력을 받아 유효성을 검사하고, RAG 검색에 적합한 여러 개의 쿼리를 생성합니다.
    """
    print("--- AGENT: Team 1 (질문 처리) 실행 ---")
    user_input = state.get("user_input", "")
    if not user_input.strip():
        return {"q_validity": False, "error_message": "입력된 질문이 없습니다."}

    parser = JsonOutputParser(p_object=QuestionProcessingResult)
    
    prompt = PromptTemplate.from_template("""
You are the first-stage agent in a RAG pipeline.

TASKS
1) q_validity: Decide if the user input is a valid, answerable question (True/False).
   - false if too vague / missing constraints / unsafe.
2) q_en_transformed: Rewrite the question into clear English (preserve domain terms, numbers, units).
3) rag_queries: Generate 2–4 short, diverse, search-friendly Korean queries (≤15 words each).
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

    llm = ChatOpenAI(
        model=config.LLM_MODEL_TEAM1,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    chain = prompt | llm | parser

    try:
        result = chain.invoke({"user_input": user_input})
        # Basic validation on the result
        if not result.get("rag_queries"):
            raise ValueError("rag_queries cannot be empty.")
        return result
    except Exception as e:
        print(f"❌ Team 1 (질문 처리) 오류: {e}")
        # Return a failure status directly
        return {"status": {"team1": "fail"}, "error_message": f"Team1 Worker: 오류 발생 - {e}"}


# --- Node 2: 질문 평가 (Evaluator) ---

class QuestionEvaluationResult(BaseModel):
    """질문 평가 노드의 LLM 결과 스키마"""
    semantic_alignment: float = Field(ge=0.0, le=1.0, description="사용자 입력과 q_en_transformed의 의미적 정합성 점수 [0,1]")
    format_compliance: bool
    rag_query_scores: List[float] = Field(default_factory=list)
    error_message: str = ""

def evaluate_question(state: GlobalState) -> Dict[str, Any]:
    """
    'process_question' 노드의 결과를 평가하고, 다음 단계로 진행할 최적의 RAG 쿼리를 선정합니다.
    """
    print("--- AGENT: Team 1 (결과 평가) 실행 ---")
    user_input = state.get("user_input", "")
    q_validity = state.get("q_validity", False)
    q_en_transformed = state.get("q_en_transformed", "")
    rag_queries = state.get("rag_queries", [])
    output_format = state.get("output_format", ["qa", "ko"])

    if not q_validity or not all([user_input, q_en_transformed, rag_queries]):
         current_retries = state.get("team1_retries", 0)
         return {
             "status": {"team1": "fail"}, 
             "error_message": "Team1 Evaluator: 평가에 필요한 정보 부족",
             "team1_retries": current_retries + 1
         }
    
    parser = JsonOutputParser(p_object=QuestionEvaluationResult)
    
    prompt = PromptTemplate.from_template("""
You are the Team1 Supervisor evaluator. Using the information below, make binary judgments and per-query scores.

[User Input]
{user_input}

[q_en_transformed]  # refined English question from the agent
{q_en_transformed}

[output_format]  # [type, language]
{output_format}

[default_format]
{default_format}

[rag_queries]
{rag_queries_json}

Criteria:
1) semantic_alignment (float in [0,1]): A continuous score for how accurately q_en_transformed reflects the meaning and constraints of user_input.
   - 1.0 = perfectly faithful; 0.0 = unrelated/incorrect.
2) format_compliance (bool): Follow these steps IN ORDER to decide:
   a) First, analyze [User Input]. Does it explicitly request a specific output format or language (e.g., "표로", "영어로", "in a table", "in English")?
   b) **If the user SPECIFIED a format:** `format_compliance` is TRUE if [output_format] correctly matches the user's request. **The [default_format] is IRRELEVANT and should be ignored in this case.**
   c) **If the user did NOT specify a format:** `format_compliance` is TRUE only if [output_format] is exactly the same as [default_format].
3) rag_query_scores (list[float]): For each rag_query, output a score in [0, 1] indicating how well it captures the user’s requirements
   (entities/keywords, constraints, time ranges, numbers/units, search-friendliness). Length MUST equal len(rag_queries).
4) error_message (str): If anything is wrong or inconsistent, write a short Korean message describing the issue; otherwise return an empty string "".

Return JSON ONLY. Do not include any additional text.

Output schema:
{schema}
""").partial(schema=parser.get_format_instructions())

    llm = ChatOpenAI(
        model=config.LLM_MODEL_TEAM1,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    chain = prompt | llm | parser

    try:
        default_format = ["qa", "ko"]
        result_dict = chain.invoke({
            "user_input": user_input,
            "q_en_transformed": q_en_transformed,
            "output_format": json.dumps(output_format, ensure_ascii=False),
            "default_format": json.dumps(default_format, ensure_ascii=False),
            "rag_queries_json": json.dumps(rag_queries, ensure_ascii=False)
        })
        result = QuestionEvaluationResult.model_validate(result_dict)

        # Python-side validation
        if len(result.rag_query_scores) != len(rag_queries):
            raise ValueError("Score list length does not match query list length.")
        if not (0.0 <= result.semantic_alignment <= 1.0):
            raise ValueError("semantic_alignment must be within [0,1].")

        passed = (result.semantic_alignment >= 0.8) and result.format_compliance
        if passed:
            # Find the best query using the scores
            best_idx = max(range(len(result.rag_query_scores)), key=lambda i: result.rag_query_scores[i])
            return {
                "status": {"team1": "pass"},
                "rag_query": rag_queries[best_idx], # Set the best query
            }
        else:
            current_retries = state.get("team1_retries", 0)
            # error_message가 비어 있고 불합격 사유가 점수 때문이라면 보조 메시지 제공
            err = result.error_message or (
                "Team1: 평가 기준 미달 (semantic_alignment < 0.8 또는 format_compliance=false)"
            )
            return {
                "status": {"team1": "fail"},
                "error_message": err,
                "team1_retries": current_retries + 1
            }
    except Exception as e:
        current_retries = state.get("team1_retries", 0)
        print(f"❌ Team 1 (결과 평가) 오류: {e}")
        return {
            "status": {"team1": "fail"}, 
            "error_message": f"Team1 Evaluator: 오류 발생 - {e}",
            "team1_retries": current_retries + 1
        }