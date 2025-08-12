# agents/supervisor_team1.py
import json
import os
from typing import List
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from agents.team1_query_agent import agent_team1_question_processing

class Team1EvalResult(BaseModel):
    semantic_alignment: bool
    format_compliance: bool
    rag_query_scores: List[float] = Field(default_factory=list)
    error_message: str = ""

eval_parser = JsonOutputParser(pydantic_object=Team1EvalResult)

EVAL_PROMPT = PromptTemplate.from_template("""
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
1) semantic_alignment (bool): Does q_en_transformed accurately reflect the meaning and constraints of user_input?
2) format_compliance (bool):
   - If user_input hints a format/language, compare that with output_format.
   - If not, check whether output_format equals default_format.
3) rag_query_scores (list[float]): For each rag_query, output a score in [0, 1] indicating how well it captures the user’s requirements
   (entities/keywords, constraints, time ranges, numbers/units, search-friendliness). Length MUST equal len(rag_queries).
4) error_message (str): If anything is wrong or inconsistent, write a short Korean message describing the issue; otherwise return an empty string "".

Return JSON ONLY. Do not include any additional text.

Output schema:
{schema}
""").partial(schema=eval_parser.get_format_instructions())

def _get_llm() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def supervisor_team1(state: dict) -> dict:
    print("🔍 Team 1 Supervisor 시작")
    state.setdefault("status", {})
    state.setdefault("error_message", "")

    default_format = ["qa", "ko"]
    max_attempts = 2
    current_attempts = 0

    while current_attempts < max_attempts:
        attempt_no = current_attempts + 1
        print(f"⚙️ Team 1 Agent 실행 (시도 {attempt_no}/{max_attempts})")
        # 필요시 merge 형태로 변경해도 됨: state.update(...)
        state = RunnableLambda(agent_team1_question_processing).invoke(state)

        if not state.get("q_validity", False):
            state["error_message"] = "Team1: 질문이 모호하거나 부적절하여 처리할 수 없습니다."
            print(f"❌ q_validity=False → 재시도: {attempt_no}/{max_attempts}")
            current_attempts += 1
            if current_attempts >= max_attempts:
                state["status"]["team1"] = "fail"
                return state
            continue

        rag_queries = state.get("rag_queries", []) or []
        if not rag_queries:
            state["error_message"] = "Team1: RAG 쿼리 후보가 생성되지 않았습니다."
            print(f"❌ rag_queries 비어있음 → 재시도: {attempt_no}/{max_attempts}")
            current_attempts += 1
            if current_attempts >= max_attempts:
                state["status"]["team1"] = "fail"
                return state
            continue

        try:
            llm = _get_llm()
            chain = EVAL_PROMPT | llm | eval_parser
            out = chain.invoke({
                "user_input": state.get("user_input", ""),
                "q_en_transformed": state.get("q_en_transformed", ""),
                "output_format": json.dumps(state.get("output_format", default_format), ensure_ascii=False),
                "default_format": json.dumps(default_format, ensure_ascii=False),
                "rag_queries_json": json.dumps(rag_queries, ensure_ascii=False),
            })

            # ✅ 반환 정규화: dict / BaseModel / str(JSON) 모두 수용
            if isinstance(out, Team1EvalResult):
                result = out
            elif isinstance(out, dict):
                result = Team1EvalResult.model_validate(out)
            elif isinstance(out, str):
                result = Team1EvalResult.model_validate(json.loads(out))
            else:
                result = Team1EvalResult.model_validate(json.loads(str(out)))

        except (ValidationError, OutputParserException) as e:
            print(f"❌ 평가 파싱 오류 (시도 {attempt_no}): {e}")
            state["error_message"] = "Team1: 평가 결과 파싱 실패"
            current_attempts += 1
            if current_attempts >= max_attempts:
                state["status"]["team1"] = "fail"
                return state
            continue
        except Exception as e:
            print(f"❌ 평가 중 예외 (시도 {attempt_no}): {e}")
            state["error_message"] = "Team1: 평가 중 알 수 없는 오류"
            current_attempts += 1
            if current_attempts >= max_attempts:
                state["status"]["team1"] = "fail"
                return state
            continue

        # 길이 정합성
        scores = list(result.rag_query_scores or [])
        if len(scores) != len(rag_queries):
            print(f"❌ 점수 길이 불일치 (시도 {attempt_no}): scores={len(scores)} vs queries={len(rag_queries)}")
            state["error_message"] = "Team1: rag_query_scores 길이가 rag_queries와 다릅니다."
            current_attempts += 1
            if current_attempts >= max_attempts:
                state["status"]["team1"] = "fail"
                return state
            continue

        # 불리언 평가
        passed = bool(state["q_validity"]) and bool(result.semantic_alignment) and bool(result.format_compliance)
        if not passed:
            reasons = []
            if not result.semantic_alignment:
                reasons.append("semantic_alignment=False")
            if not result.format_compliance:
                reasons.append("format_compliance=False")
            model_msg = (result.error_message or "").strip()
            state["error_message"] = model_msg if model_msg else f"Team1: 평가 실패({', '.join(reasons)})"
            print(f"🔁 평가 실패 → 재시도: {attempt_no}/{max_attempts}")
            current_attempts += 1
            if current_attempts >= max_attempts:
                state["status"]["team1"] = "fail"
                return state
            continue

        # 통과: best rag query 선택
        best_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
        state["rag_query"] = rag_queries[best_idx]
        state["rag_query_scores"] = scores
        state["status"]["team1"] = "pass"
        state["error_message"] = ""
        print(f"✅ Team 1 통과 / best_rag_query idx={best_idx}, score={scores[best_idx]:.2f}")
        return state

    state["status"]["team1"] = "fail"
    if not state.get("error_message"):
        state["error_message"] = "Team1: 알 수 없는 이유로 실패했습니다."
    return state
