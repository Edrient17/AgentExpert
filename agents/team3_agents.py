# agents/team3_agents.py

import json
from typing import Dict, Any

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

import config
from state import GlobalState
from utility_tools import format_docs

# --- Node 1: 답변 생성 (Worker) ---

DOCS_ANSWER_PROMPT = PromptTemplate.from_template("""
You are the Team 3 answer generator in a Multi-Agent Q&A pipeline.
Your primary task is to synthesize an answer based on the provided passages.

You will receive:
- A refined English question
- Retrieved passages as a concatenated preview (may be truncated)

Your job:
- Produce the final answer strictly in the requested output format and language.

<think>
Passages:
{passages}

I will use both the information from these passages and my own prior knowledge to answer the question.
First, I will carefully examine whether the passages contain any misinformation, contradictions, or irrelevant details.
Then, I will combine verified facts from the passages with only safe, generally accepted prior knowledge to derive the correct answer.
If conflicts remain or the evidence is insufficient, I will output the no-information message instead of guessing.
I will NOT reveal any reasoning or the content of this <think> block in the final output.
</think>

Requested format: {out_type}
Requested language: {answer_language}

Format guidelines (flexible within type):
- qa:
  • Begin with a one-sentence direct answer.
  • Then provide ample explanation (multiple short paragraphs and/or 3–8 bullets).
  • If a procedure is involved, include a numbered step list. May add "Edge cases", "Tips" sections if useful.
- bulleted:
  • 8–15 bullets with meaningful substance (≤40 words each).
  • You may group bullets by mini-headings (bold) and use one level of sub-bullets when needed.
- table:
  • Produce a Markdown table with a header row; derive 3–9 sensible columns from the question/context.
  • Include as many rows as needed (up to ~100). Add a short "Notes" paragraph below if clarification helps.
  • Use "N/A" for missing values.
- json:
  • Return valid JSON ONLY (no code fences).
  • Always include an "answer" string. You may add keys like "evidence", "steps", "assumptions", "limitations", "confidence" (0–1), etc.
  • Keep keys concise; use arrays/objects as needed.
- report:
  • Design your own section plan (3–10 H2 sections) tailored to the question.
  • Each section 3–8 sentences; include lists/tables where helpful. Subsections (H3) allowed.

Grounding & safety rules:
- Use the passages as primary evidence. You MAY use prior knowledge only if it does not contradict the passages.
- If the passages are insufficient or conflicting, respond with the no-information message in the requested language and format:
  - ko: "문서에 해당 정보가 없습니다."
  - en: "The documents do not contain that information."
- Do NOT add any prefixes about requested format prior to the answer.
- Do NOT invent or hallucinate facts.
- Do NOT include citations or URLs unless they explicitly appear in the passages.
- DO NOT reveal the content of <think> or any reasoning steps.

Write STRICTLY in: {answer_language}

Inputs:
[Refined question]
{q_en_transformed}

Answer:
""")

GENERAL_ANSWER_PROMPT = PromptTemplate.from_template("""
You are a helpful AI assistant. Your task is to answer the user's question using your own internal knowledge.
There are no external documents provided for this question.

Your job:
- Produce the final answer strictly in the requested output format and language.

Requested format: {out_type}
Requested language: {answer_language}

Grounding & safety rules:
- Do NOT add any prefixes about requested format prior to the answer.
                                                                                                                                                         
Format guidelines (flexible within type):
- qa:
  • Begin with a one-sentence direct answer.
  • Then provide ample explanation (multiple short paragraphs and/or 3–8 bullets).
  • If a procedure is involved, include a numbered step list. May add "Edge cases", "Tips" sections if useful.
- bulleted:
  • 8–15 bullets with meaningful substance (≤40 words each).
  • You may group bullets by mini-headings (bold) and use one level of sub-bullets when needed.
- table:
  • Produce a Markdown table with a header row; derive 3–9 sensible columns from the question/context.
  • Include as many rows as needed (up to ~100). Add a short "Notes" paragraph below if clarification helps.
  • Use "N/A" for missing values.
- json:
  • Return valid JSON ONLY (no code fences).
  • Always include an "answer" string. You may add keys like "evidence", "steps", "assumptions", "limitations", "confidence" (0–1), etc.
  • Keep keys concise; use arrays/objects as needed.
- report:
  • Design your own section plan (3–10 H2 sections) tailored to the question.
  • Each section 3–8 sentences; include lists/tables where helpful. Subsections (H3) allowed.

Write STRICTLY in: {answer_language}

Inputs:
[Refined question]
{q_en_transformed}

Answer:
""")

def generate_answer(state: GlobalState) -> Dict[str, Any]:
    """
    Team 1, 2에서 정제된 데이터를 바탕으로 최종 답변을 생성합니다.
    """
    print("--- AGENT: Team 3 (답변 생성) 실행 ---")
    question = state.get("q_en_transformed", "")
    output_format = state.get("output_format", ["qa", "ko"])
    rag_docs = state.get("rag_docs", [])
    web_docs = state.get("web_docs", [])

    docs = web_docs if web_docs else rag_docs
    out_type, answer_language = output_format[0], output_format[1]

    if docs:
        print("... 문서를 기반으로 답변 생성")
        prompt = DOCS_ANSWER_PROMPT
        invoke_params = {
            "q_en_transformed": question,
            "passages": format_docs(docs),
            "out_type": out_type,
            "answer_language": answer_language,
        }
    else:
        print("... LLM 자체 지식으로 답변 생성")
        prompt = GENERAL_ANSWER_PROMPT
        invoke_params = {
            "q_en_transformed": question,
            "out_type": out_type,
            "answer_language": answer_language,
        }

    llm = ChatOpenAI(model=config.LLM_MODEL_TEAM3, temperature=0.0)
    chain = prompt | llm

    try:
        result = chain.invoke(invoke_params)
        return {"generated_answer": result.content.strip()}
    except Exception as e:
        print(f"❌ Team 3 (답변 생성) 오류: {e}")
        return {"status": {"team3": "fail"}, "error_message": f"Team3 Worker: 오류 발생 - {e}"}

# --- Node 2: 답변 평가 (Evaluator) ---

class AnswerEvaluationResult(BaseModel):
    """답변 평가 노드의 LLM 결과 스키마"""
    rules_compliance: bool
    question_coverage: float
    logical_structure: float
    error_message: str = ""

def evaluate_answer(state: GlobalState) -> Dict[str, Any]:
    """
    'generate_answer' 노드의 결과를 평가하여 최종적으로 사용자에게 전달될 품질인지 검수합니다.
    """
    print("--- AGENT: Team 3 (답변 평가) 실행 ---")
    question = state.get("q_en_transformed", "")
    output_format = state.get("output_format", ["qa", "ko"])
    answer = state.get("generated_answer", "")

    if not all([question, output_format, answer]):
        current_retries = state.get("team3_retries", 0)
        return {
            "status": {"team3": "fail"}, 
            "error_message": "Team3 Evaluator: 평가에 필요한 정보 부족",
            "team3_retries": current_retries + 1
        }
    
    parser = JsonOutputParser(p_object=AnswerEvaluationResult)
    
    prompt = PromptTemplate.from_template("""
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
   - type ∈ ["qa","bulleted","table","json","report"]
   - language ∈ ["ko","en"]: The answer must be in the requested language.
2) question_coverage (float): Score from 0.0 to 1.0. How well does the answer address the refined question (intent, scope, constraints)?
3) logical_structure (float): Score from 0.0 to 1.0. How coherent and logically well-structured is the answer?

Return JSON ONLY with:
{schema}
""").partial(schema=parser.get_format_instructions())

    llm = ChatOpenAI(
        model=config.LLM_MODEL_TEAM3,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    chain = prompt | llm | parser

    try:
        result_dict = chain.invoke({
            "q_en_transformed": question,
            "output_format": json.dumps(output_format, ensure_ascii=False),
            "generated_answer": answer
        })
        result = AnswerEvaluationResult.model_validate(result_dict)

        # 세 가지 평가 기준을 모두 통과해야 최종 'pass'
        passed = (
            result.rules_compliance and
            result.question_coverage >= 0.7 and
            result.logical_structure >= 0.7
        )

        if passed:
            return {"status": {"team3": "pass"}}
        else:
            current_retries = state.get("team3_retries", 0)

            reasons = []
            if not result.rules_compliance:
                reasons.append("format")
            if result.question_coverage < 0.7:
                reasons.append(f"coverage({result.question_coverage:.2f})")
            if result.logical_structure < 0.7:
                reasons.append(f"logic({result.logical_structure:.2f})")
            
            error_msg = result.error_message or f"Team3: 답변 품질 미달 ({', '.join(reasons)})"
            return {
                "status": {"team3": "fail"}, 
                "error_message": error_msg,
                "team3_retries": current_retries + 1
            }
    except Exception as e:
        current_retries = state.get("team3_retries", 0)
        print(f"❌ Team 3 (답변 평가) 오류: {e}")
        return {
            "status": {"team3": "fail"}, 
            "error_message": f"Team3 Evaluator: 오류 발생 - {e}",
            "team3_retries": current_retries + 1
        }