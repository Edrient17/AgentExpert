
# agents/team1_agents.py

import json
import uuid
from typing import List, Dict, Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from pydantic import BaseModel, Field

import config
from state import AgentState

# --- Pydantic 스키마 (변경 없음) ---
class QuestionProcessingResult(BaseModel):
    q_validity: bool = Field(description="사용자 질문이 답변 가능한 유효한 질문인지 여부 (True/False)")
    q_en_transformed: str = Field(description="사용자 질문을 명확하게 재구성한 영문 질문")
    rag_queries: List[str] = Field(description="검색에 사용할 2-4개의 다양한 영문 RAG 쿼리 후보 리스트", min_items=2, max_items=4)
    output_format: List[str] = Field(description="요청된 출력 포맷 [type, language]", min_items=2, max_items=2)

class QuestionEvaluationResult(BaseModel):
    semantic_alignment: float = Field(ge=0.0, le=1.0, description="사용자 입력과 q_en_transformed의 의미적 정합성 점수 [0,1]")
    format_compliance: bool
    rag_query_scores: List[float] = Field(default_factory=list)
    error_message: str = ""

# --- Node 1: 질문 처리 (Worker) ---
def process_question(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Team 1 (질문 처리) 실행 ---")

    manager_feedback = state.get("manager_feedback")
    last_message = state['messages'][-1]

    feedback_instructions = ""
    if manager_feedback:
        print(f"📝 매니저 피드백 수신: {manager_feedback}")
        feedback_instructions = f"""
        **IMPORTANT REVISION INSTRUCTION FROM MANAGER:**
        The previous attempt was not good enough. You MUST incorporate the following feedback into your new result:
        "{manager_feedback}"
        """
        state["manager_feedback"] = None

    # Check for internal team feedback from the evaluator
    if isinstance(last_message, ToolMessage) and last_message.name == "team1_evaluator" and last_message.content.startswith("retry"):
        internal_feedback = last_message.content.replace("retry:", "").strip()
        if internal_feedback:
            print(f"📝 팀 내부 피드백 수신: {internal_feedback}")
            feedback_instructions += f"""
            **IMPORTANT INTERNAL FEEDBACK FOR REVISION:**
            Your previous attempt failed the internal quality check. You MUST address the following issue:
            "{internal_feedback}"
            """
        
    user_input = next((msg.content for msg in reversed(state['messages']) if isinstance(msg, HumanMessage)), "")
    if not user_input.strip():
        return {"messages": [ToolMessage(content="fail: 입력된 질문이 없습니다.", name="team1_worker", tool_call_id=str(uuid.uuid4()))]}

    parser = JsonOutputParser(p_object=QuestionProcessingResult)
    prompt = PromptTemplate.from_template("""
You are the first-stage agent in a RAG pipeline.

TASKS
1) q_validity: Decide if the user input is a valid, answerable question (True/False).
   - false if too vague / missing constraints / unsafe.
2) q_en_transformed: Rewrite the question into clear English (preserve domain terms, numbers, units).
3) rag_queries: Generate 2-4 concise and keyword-focused search queries. Each query should extract the essential nouns and core intent from the user's input.
   - Mix styles (keyword, semantic paraphrase, entity-focused, time-bounded) when applicable.
   - Do NOT invent facts not implied by the user input. Return 2–4 items only.
4) output_format: ALWAYS return a 2-item array [type, language].
   - type ∈ ["report","table","bulleted","qa"]
   - language ∈ ["ko","en"]
   - Defaults apply independently:
       • If type is missing/unclear/invalid → use "qa".
       • If language is missing/unclear/invalid → use "ko".
   - If only one of (type, language) can be inferred, fill the other with its default.
   - Normalize to lowercase. Return exactly two items, no more, no less.

{feedback_instructions}
                                          
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
    chain = prompt.partial(feedback_instructions=feedback_instructions, schema=parser.get_format_instructions()) | llm | parser

    try:
        result_dict = chain.invoke({"user_input": user_input})
        if not result_dict.get("rag_queries"):
            raise ValueError("rag_queries가 비어있습니다.")
        return {
            "user_input": user_input,
            "q_en_transformed": result_dict.get("q_en_transformed", ""),
            "messages": [
                AIMessage(
                    content="사용자 질문을 성공적으로 분석했습니다.",
                    additional_kwargs=result_dict
                )
            ]
        }
    except Exception as e:
        print(f"❌ Team 1 (질문 처리) 오류: {e}")
        return {"messages": [ToolMessage(content=f"fail: Team1 Worker 오류 - {e}", name="team1_worker", tool_call_id=str(uuid.uuid4()))]}

# --- Node 2: 질문 평가 (Evaluator) ---
def evaluate_question(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Team 1 (결과 평가) 실행 ---")
    last_message = state['messages'][-1]
    if not isinstance(last_message, AIMessage) or not last_message.additional_kwargs:
        return {"messages": [ToolMessage(content="fail: Team1 평가자가 분석 결과를 찾을 수 없습니다.", name="team1_evaluator", tool_call_id=str(uuid.uuid4()))]}

    current_retries = state.get("team1_retries", 0)
    state["team1_retries"] = current_retries + 1

    processed_data = last_message.additional_kwargs
    user_input = next((msg.content for msg in state['messages'] if isinstance(msg, HumanMessage)), "")

    q_validity = processed_data.get("q_validity", False)
    q_en_transformed = processed_data.get("q_en_transformed", "")
    rag_queries = processed_data.get("rag_queries", [])
    output_format = processed_data.get("output_format", ["qa", "ko"])
    
    if not q_validity or not all([user_input, q_en_transformed, rag_queries]):
        return {"messages": [ToolMessage(content="fail: 평가에 필요한 정보 부족", name="team1_evaluator", tool_call_id=str(uuid.uuid4()))]}
    
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
""").partial(schema=parser.get_format_instructions()) # 프롬프트 내용은 기존과 동일
    llm = ChatOpenAI(
        model=config.LLM_MODEL_TEAM1,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    chain = prompt | llm | parser

    try:
        result_dict = chain.invoke({
            "user_input": user_input,
            "q_en_transformed": q_en_transformed,
            "output_format": json.dumps(output_format, ensure_ascii=False),
            "default_format": json.dumps(["qa", "ko"], ensure_ascii=False),
            "rag_queries_json": json.dumps(rag_queries, ensure_ascii=False)
        })
        result = QuestionEvaluationResult.model_validate(result_dict)

        if len(result.rag_query_scores) != len(rag_queries):
            raise ValueError("점수 리스트와 쿼리 리스트의 길이가 다릅니다.")
        
        passed = (result.semantic_alignment >= 0.8) and result.format_compliance
        if passed:
            best_idx = max(range(len(result.rag_query_scores)), key=lambda i: result.rag_query_scores[i])
            best_query = rag_queries[best_idx]
            
            return {
                # ⬇️ 상태에 직접 저장
                "best_rag_query": best_query,
                "q_en_transformed": q_en_transformed,
                "messages": [
                    ToolMessage(
                        content="pass",
                        name="team1_evaluator",
                        tool_call_id=str(uuid.uuid4()),
                        additional_kwargs={
                            "q_en_transformed": q_en_transformed,
                            "output_format": output_format,
                            "best_rag_query": best_query,
                        }
                    )
                ]
            }
        else:
            err = result.error_message or "Team1: 평가 기준 미달 (Team1: Evaluation criteria not met)"
            if current_retries < config.MAX_RETRIES_TEAM1:
                print(f"🔁 Team 1 평가 실패. 재시도를 요청합니다. ({current_retries + 1}/{config.MAX_RETRIES_TEAM1})")
                return {"messages": [ToolMessage(content=f"retry: {err}", name="team1_evaluator", tool_call_id=str(uuid.uuid4()))]}
            else:
                print(f"❌ Team 1 최종 실패 (재시도 {config.MAX_RETRIES_TEAM1}회 초과).")
                return {"messages": [ToolMessage(content=f"fail: {err}", name="team1_evaluator", tool_call_id=str(uuid.uuid4()))]}
             
    except Exception as e:
        print(f"❌ Team 1 (결과 평가) 오류: {e}")
        if current_retries < config.MAX_RETRIES_TEAM1:
             return {"messages": [ToolMessage(content="retry", name="team1_evaluator", tool_call_id=str(uuid.uuid4()))]}
        else:
             return {"messages": [ToolMessage(content=f"fail: Team1 Evaluator 오류 - {e}", name="team1_evaluator", tool_call_id=str(uuid.uuid4()))]}
