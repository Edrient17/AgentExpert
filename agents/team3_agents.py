# agents/team3_agents.py
import json
import uuid
from typing import Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

import config
from state import AgentState
from utility_tools import format_docs

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

Before finalizing the answer, I will double-check that every single statement in my generated answer is directly supported by the provided passages. If I cannot find direct evidence for a piece of information, I will remove it from the answer.

If conflicts remain or the evidence is insufficient after this check, I will output the no-information message instead of guessing.
I will NOT reveal any reasoning or the content of this <think> block in the final output.
</think>

{feedback_instructions}
                                                  
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

# --- Pydantic 스키마 ---
class AnswerEvaluationResult(BaseModel):
    rules_compliance: bool
    question_coverage: float
    logical_structure: float
    hallucination_score: float
    error_message: str = ""

def _get_context_from_history(state: AgentState) -> dict:
    """메시지 히스토리에서 답변 생성에 필요한 모든 컨텍스트를 추출합니다."""
    context = {
        "q_en_transformed": "",
        "output_format": ["qa", "ko"],
        "docs": []
    }
    for msg in reversed(state['messages']):
        if isinstance(msg, ToolMessage) and msg.name == "team2_evaluator" and msg.content == "pass":
            context["docs"] = msg.additional_kwargs.get("retrieved_docs", [])
        
        if not context["q_en_transformed"] and isinstance(msg, ToolMessage) and msg.name == "team1_evaluator":
            context["q_en_transformed"] = msg.additional_kwargs.get("q_en_transformed", "")
            context["output_format"] = msg.additional_kwargs.get("output_format", ["qa", "ko"])
    
    return context

# --- Node 1: 답변 생성 (Worker) ---
def generate_answer(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Team 3 (답변 생성) 실행 ---")

    manager_feedback = state.get("manager_feedback")
    last_message = state['messages'][-1]

    feedback_instructions = ""
    if manager_feedback:
        print(f"📝 매니저 피드백 수신 (Team 3): {manager_feedback}")
        feedback_instructions = f"""
**IMPORTANT REVISION INSTRUCTION FROM MANAGER:**
Your previous answer was not satisfactory. You MUST revise your answer based on the following feedback:
"{manager_feedback}"
"""
        state["manager_feedback"] = None

    if isinstance(last_message, ToolMessage) and last_message.name == "final_evaluator" and last_message.content.startswith("retry"):
        internal_feedback = last_message.content.replace("retry:", "").strip()
        if internal_feedback:
            print(f"📝 팀 내부 피드백 수신 (Team 3): {internal_feedback}")
            feedback_instructions += f"""
            **IMPORTANT INTERNAL FEEDBACK FOR REVISION:**
            Your previous answer failed the internal quality check. You MUST revise your answer based on the following issue:
            "{internal_feedback}"
            """

    context = _get_context_from_history(state)
    
    question = context["q_en_transformed"]
    output_format = context["output_format"]
    docs = context["docs"]
    out_type, answer_language = output_format[0], output_format[1]

    if docs:
        print("... 문서를 기반으로 답변 생성")
        prompt = DOCS_ANSWER_PROMPT # DOCS_ANSWER_PROMPT 기존 내용
        invoke_params = {
            "q_en_transformed": question,
            "passages": format_docs(docs),
            "out_type": out_type,
            "answer_language": answer_language,
        }
    else:
        print("... LLM 자체 지식으로 답변 생성")
        prompt = GENERAL_ANSWER_PROMPT # GENERAL_ANSWER_PROMPT 기존 내용
        invoke_params = {
            "q_en_transformed": question,
            "out_type": out_type,
            "answer_language": answer_language,
        }

    llm = ChatOpenAI(model=config.LLM_MODEL_TEAM3, temperature=0.0)
    chain = prompt.partial(feedback_instructions=feedback_instructions) | llm

    try:
        result = chain.invoke(invoke_params)
        return {"messages": [AIMessage(content=result.content.strip())]}
    except Exception as e:
        print(f"❌ Team 3 (답변 생성) 오류: {e}")
        return {"messages": [ToolMessage(content=f"fail: Team3 Worker 오류 - {e}", name="team3_worker", tool_call_id=str(uuid.uuid4()))]}

# --- Node 2: 답변 평가 (Evaluator) ---
def evaluate_answer(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Team 3 (답변 평가) 실행 ---")
    generated_answer_msg = state['messages'][-1]
    if not isinstance(generated_answer_msg, AIMessage):
        return {"messages": [ToolMessage(content="fail: 평가할 답변을 찾을 수 없습니다.", name="team3_evaluator", tool_call_id=str(uuid.uuid4()))]}
    
    current_retries = state.get("team3_retries", 0)
    state["team3_retries"] = current_retries + 1
    
    answer = generated_answer_msg.content
    context = _get_context_from_history(state)
    question = context["q_en_transformed"]
    output_format = context["output_format"]
    
    if not all([question, output_format, answer]):
        return {"messages": [ToolMessage(content="fail: 평가에 필요한 정보 부족", name="team3_evaluator", tool_call_id=str(uuid.uuid4()))]}
    
    parser = JsonOutputParser(p_object=AnswerEvaluationResult)
    prompt = PromptTemplate.from_template("""
You are the Team 3 Supervisor evaluator. Judge the final answer against the requested format,
the refined question, AND the provided documents.

Inputs:
[Refined question]
{q_en_transformed}

[Output format]  # ["type", "language"]
{output_format}

[Generated answer]
{generated_answer}

[Retrieved docs]
{retrieved_docs}

Criteria:
1) rules_compliance (bool): Does the answer follow the requested output_format?
   - type ∈ ["qa","bulleted","table","json","report"]
   - language ∈ ["ko","en"]: The answer must be in the requested language.
2) question_coverage (float): Score from 0.0 to 1.0. How well does the answer address the refined question (intent, scope, constraints)?
3) logical_structure (float): Score from 0.0 to 1.0. How coherent and logically well-structured is the answer?
4) hallucination_score (float): 0.0–1.0. To what extent is the answer grounded in the retrieved docs?
   - 1.0 = entirely grounded, 0.0 = completely hallucinated.
                                          
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
            "generated_answer": answer,
            "retrieved_docs": format_docs(context["docs"])
        })
        result = AnswerEvaluationResult.model_validate(result_dict)

        passed = (
            result.rules_compliance and
            result.question_coverage >= 0.7 and
            result.logical_structure >= 0.7 and
            result.hallucination_score >= 0.7
        )

        if passed:
            return {"messages": [ToolMessage(content="pass", name="final_evaluator", tool_call_id=str(uuid.uuid4()))]}
        else:
            if current_retries < config.MAX_RETRIES_TEAM3:
                print(f"🔁 Team 3 평가 실패. 재시도를 요청합니다. ({current_retries + 1}/{config.MAX_RETRIES_TEAM3})")
                err = result.error_message or "답변 품질 미달 (Answer quality is insufficient)"
                return {"messages": [ToolMessage(content=f"retry: {err}", name="final_evaluator", tool_call_id=str(uuid.uuid4()))]}
            else:
                print(f"❌ Team 3 최종 실패 (재시도 {config.MAX_RETRIES_TEAM3}회 초과).")
                return {"messages": [ToolMessage(content="fail: 답변 품질 미달", name="final_evaluator", tool_call_id=str(uuid.uuid4()))]}
           
    except Exception as e:
        print(f"❌ Team 3 (답변 평가) 오류: {e}")
        if current_retries < config.MAX_RETRIES_TEAM3:
            return {"messages": [ToolMessage(content="retry", name="final_evaluator", tool_call_id=str(uuid.uuid4()))]}
        else:
            return {"messages": [ToolMessage(content=f"fail: Team3 Evaluator 오류 - {e}", name="final_evaluator", tool_call_id=str(uuid.uuid4()))]}
