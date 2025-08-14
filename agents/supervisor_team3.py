# agents/supervisor_team3.py
import os
import json
from typing import Dict
from pydantic import BaseModel, ValidationError

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from utils import get_llm

from agents.team3_answer_agent import agent_team3_answer_generation

# ===== Eval schema =====
class Team3EvalResult(BaseModel):
    rules_compliance: bool       # respects output_format type & language
    question_coverage: bool      # answers the refined question appropriately
    logical_structure: bool      # coherent, well-structured reasoning/flow
    error_message: str = ""

eval_parser = JsonOutputParser(pydantic_object=Team3EvalResult)

# ===== Evaluation prompt =====
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
     * qa: Starts with a one-sentence direct answer. This is followed by a helpful, supplementary explanation which can be in the form of bullet points, paragraphs, or a numbered list. The length and detail of the explanation are flexible as long as they are relevant.
     * table: Markdown table with a header row; 3–7 sensible columns; "N/A" for missing values.
     * json: Valid JSON ONLY (no code fences). Concise keys; strings/numbers/arrays/objects only.
     * report: Markdown H2 sections in order: "## Summary", "## Findings", "## Method", "## Limitations".
   - language ∈ ["ko","en"]: The entire answer must be in the requested language.
2) question_coverage (bool): Does the answer appropriately address the refined question (intent, scope, constraints)?
3) logical_structure (bool): Is the answer coherent and logically well-structured for a reliable response?

Return JSON ONLY with:
{schema}
""").partial(schema=eval_parser.get_format_instructions())

# ===== Supervisor =====
def supervisor_team3(state: Dict) -> Dict:
    print("🔍 Team 3 Supervisor 시작")
    state.setdefault("status", {})
    state.setdefault("error_message", "")

    max_attempts = 2
    attempt = 0
    
    llm = get_llm()

    while attempt < max_attempts:
        attempt += 1
        print(f"⚙️ Team 3 Agent 실행 (시도 {attempt}/{max_attempts})")
        
        state = RunnableLambda(agent_team3_answer_generation).invoke(state)

        question = (state.get("q_en_transformed") or "").strip()
        answer = (state.get("generated_answer") or "").strip()
        output_format = state.get("output_format", ["qa", "ko"]) or ["qa", "ko"]

        if not question or not answer:
            print("❌ 질문 또는 답변 누락")
            state["error_message"] = "Team3: Question or answer is empty."
            if attempt >= max_attempts:
                state["status"]["team3"] = "fail"
                return state
            continue

        try:
            chain = EVAL_PROMPT | llm | eval_parser
            out = chain.invoke({
                "q_en_transformed": question,
                "output_format": json.dumps(output_format, ensure_ascii=False),
                "generated_answer": answer,
            })

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
            state["error_message"] = "Team3: Evaluation result parsing failed."
            if attempt >= max_attempts:
                state["status"]["team3"] = "fail"
                return state
            continue
        except Exception as e:
            print(f"❌ 평가 중 예외 (시도 {attempt}): {e}")
            state["error_message"] = "Team3: Unknown error during evaluation."
            if attempt >= max_attempts:
                state["status"]["team3"] = "fail"
                return state
            continue

        passed = result.rules_compliance and result.question_coverage and result.logical_structure
        if passed:
            print("✅ Team 3 통과")
            state["status"]["team3"] = "pass"
            state["error_message"] = ""
            state["next_node"] = "end" 
            return state

        reasons = []
        if not result.rules_compliance:
            reasons.append("rules_compliance=False")
        if not result.question_coverage:
            reasons.append("question_coverage=False")
        if not result.logical_structure:
            reasons.append("logical_structure=False")

        state["error_message"] = (result.error_message or f"Team3: Evaluation failed({', '.join(reasons)})").strip()
        print(f"🔁 평가 실패 → 재시도: {attempt}/{max_attempts}")

    print("❌ 최종 실패 → Team 3 fail 처리")
    state["status"]["team3"] = "fail"
    if not state.get("error_message"):
        state["error_message"] = "Team3: Failed for an unknown reason."
    return state