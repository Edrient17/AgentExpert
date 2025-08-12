# agents/supervisor_team3.py
import os
import json
from typing import Dict
from pydantic import BaseModel, ValidationError
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from agents.team3_answer_agent import agent_team3_answer_generation

# ===== Eval schema: 3 booleans =====
class Team3EvalResult(BaseModel):
    rules_compliance: bool       # respects output_format type & language
    question_coverage: bool      # answers the refined question appropriately
    logical_structure: bool      # coherent, well-structured reasoning/flow
    error_message: str = "" 

eval_parser = JsonOutputParser(pydantic_object=Team3EvalResult)

# ===== Evaluation prompt (EN) =====
EVAL_PROMPT = PromptTemplate.from_template("""
You are the Team 3 Supervisor evaluator. Judge the final answer against the requested format and the question.

Inputs:
[Refined question]
{q_en_transformed}

[Output format]  # ["type", "language"]
{output_format}

[Generated answer]
{generated_answer}

Criteria:
1) rules_compliance (bool): Does the answer follow the requested output_format?
   - type ∈ ["qa","bulleted","table","json","report"]:
     * qa: One-sentence direct answer + 2–4 short bullets (evidence/caveats). Steps as a numbered list if procedure requested.
     * bulleted: 5–10 bullet points, one sentence each. One sub-bullet level allowed if essential.
     * table: Markdown table with a header row; 3–7 sensible columns; "N/A" for missing values.
     * json: Valid JSON ONLY (no code fences). Concise keys; strings/numbers/arrays/objects only.
     * report: Markdown H2 sections in order: "## Summary", "## Findings", "## Method", "## Limitations".
   - language ∈ ["ko","en"]: The entire answer must be written in the requested language.
2) question_coverage (bool): Does the answer appropriately address the refined question (intent, scope, constraints)?
3) logical_structure (bool): Is the answer coherent and logically well-structured for a reliable response?

Return JSON ONLY with:
{schema}
""").partial(schema=eval_parser.get_format_instructions())

# ===== Lazy LLM =====
def _get_llm() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ===== Supervisor =====
def supervisor_team3(state: Dict) -> Dict:
    print("🔍 Team 3 Supervisor 시작")
    state.setdefault("status", {})
    state.setdefault("error_message", "")

    max_attempts = 2
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        print(f"⚙️ Team 3 Agent 실행 (시도 {attempt}/{max_attempts})")
        
        state = RunnableLambda(agent_team3_answer_generation).invoke(state)

        question = (state.get("q_en_transformed") or "").strip()
        answer = (state.get("generated_answer") or "").strip()
        output_format = state.get("output_format", ["qa", "ko"]) or ["qa", "ko"]

        if not question or not answer:
            print("❌ 질문 또는 답변 누락")
            state["error_message"] = "Team3: 질문 또는 답변이 비어 있습니다."
            if attempt >= max_attempts:
                state["status"]["team3"] = "fail"
                return state
            continue

        try:
            llm = _get_llm()
            chain = EVAL_PROMPT | llm | eval_parser
            out = chain.invoke({
                "q_en_transformed": question,
                "output_format": json.dumps(output_format, ensure_ascii=False),
                "generated_answer": answer,
            })

            # 반환 정규화
            if isinstance(out, Team3EvalResult):
                result = out
            elif isinstance(out, dict):
                result = Team3EvalResult.model_validate(out)
            elif isinstance(out, str):
                result = Team3EvalResult.model_validate(json.loads(out))
            else:
                result = Team3EvalResult.model_validate(json.loads(str(out)))

        except (ValidationError, OutputParserException) as e:
            print(f"❌ 평가 파싱 오류 (시도 {attempt}): {e}")
            state["error_message"] = "Team3: 평가 결과 파싱 실패"
            if attempt >= max_attempts:
                state["status"]["team3"] = "fail"
                return state
            continue
        except Exception as e:
            print(f"❌ 평가 중 예외 (시도 {attempt}): {e}")
            state["error_message"] = "Team3: 평가 중 알 수 없는 오류"
            if attempt >= max_attempts:
                state["status"]["team3"] = "fail"
                return state
            continue

        # 최종 판정
        passed = result.rules_compliance and result.question_coverage and result.logical_structure
        if passed:
            print("✅ Team 3 통과")
            state["status"]["team3"] = "pass"
            state["error_message"] = ""
            return state

        reasons = []
        if not result.rules_compliance:
            reasons.append("rules_compliance=False")
        if not result.question_coverage:
            reasons.append("question_coverage=False")
        if not result.logical_structure:
            reasons.append("logical_structure=False")

        state["error_message"] = (result.error_message or f"Team3: 평가 실패({', '.join(reasons)})").strip()
        print(f"🔁 평가 실패 → 재시도: {attempt}/{max_attempts}")

    # 모든 시도 실패
    print("❌ 최종 실패 → Team 3 fail 처리")
    state["status"]["team3"] = "fail"
    if not state.get("error_message"):
        state["error_message"] = "Team3: 알 수 없는 이유로 실패했습니다."
    return state
