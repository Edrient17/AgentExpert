# graphs/super_graph.py

from typing import Literal, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, HumanMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

import config
from state import AgentState

# --- 매니저 에이전트의 결정을 위한 Pydantic 스키마 ---
class ManagerDecision(BaseModel):
    """매니저의 결정 스키마"""
    next_team: Literal["team1", "team2", "team3", "end"] = Field(description="다음에 작업을 수행할 팀 또는 워크플로우 종료 여부")
    feedback: Optional[str] = Field(description="작업을 수정해야 할 경우, 해당 팀에게 전달할 구체적인 한글 피드백")
    reason: str = Field(description="결정에 대한 간단한 한글 요약")

# --- 슈퍼그래프의 노드들 ---

def manager_agent(state: AgentState) -> dict:

    print("--- MANAGER: 작업 검토 및 다음 단계 결정 ---")

    global_loop_count = state.get("global_loop_count", 0)
    last_message = state['messages'][-1]

    last_name = getattr(last_message, 'name', 'N/A')
    last_content = getattr(last_message, 'content', '')

    user_question = next((msg.content for msg in state['messages'] if isinstance(msg, HumanMessage)), "")

    try:
        if last_name == "team1_evaluator" and str(last_content).strip() == "pass":
            is_simple = state.get("is_simple_query", "No")
            if is_simple == "Yes":
                print("🧭 Manager 단축 라우팅: 간단질문 → Team3 직행")
                return {
                    "next_team_to_call": "team3",
                    "manager_feedback": None,
                    "global_loop_count": global_loop_count,
                    "team3_retries": 0
                }
            else:
                print("🧭 Manager 단축 라우팅: 일반질문 → Team2 탐색")
                return {
                    "next_team_to_call": "team2",
                    "manager_feedback": None,
                    "global_loop_count": global_loop_count,
                    "team2_retries": 0
                }
    except Exception as e:
        print(f"⚠️ is_simple_query 기반 라우팅 실패: {e} (기본 LLM 라우팅으로 진행)")

    parser = JsonOutputParser(p_object=ManagerDecision)
    prompt = PromptTemplate.from_template("""
You are the project manager of a multi-agent RAG system. Your role is to review the work of your teams (Team1, Team2, Team3) and decide the next step with surgical precision.

**CONTEXT:**
- User's original question: "{user_question}"
- The last message from the previous team:
  - Team/Node: "{last_message_name}"
  - Content/Result: "{last_message_content}"

**YOUR TASK:**
Based on the context, decide the next team to call or to end the process. You can also provide feedback to a team to ask for revisions. The goal is to solve the problem by addressing the root cause.

**DECISION LOGIC (Follow these rules STRICTLY):**
Your decision MUST be based on the `last_message_name` and its `content`.

- **WHEN `last_message_name` is "team1_evaluator":**
  - If `content` is "pass", call `team2`.
  - If `content` is "fail", call `team1` with feedback.

- **WHEN `last_message_name` is "team2_evaluator":**
  - If `content` is "pass", call `team3`.
  - If `content` is "fail", call `team1` with feedback to fix the search queries. This is a critical step; poor documents usually mean poor queries.

- **WHEN `last_message_name` is "team3_evaluator":**
  - If `content` is "pass", the entire process is complete. Call `end`.
  - If `content` is "fail", call `team3` again with feedback for revision.

- **IF a tool failure is reported in the last message:**
  - The process cannot continue. Call `end` and provide a reason.

**OUTPUT (JSON ONLY):**
Provide your decision in the following JSON format.
{schema}
""").partial(schema=parser.get_format_instructions())

    llm = ChatOpenAI(model=config.LLM_MODEL_SUPER_ROUTER, temperature=0.0, model_kwargs={"response_format": {"type": "json_object"}})
    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            "user_question": user_question,
            "last_message_name": last_name,
            "last_message_content": last_content
        })
        
        next_team = result.get("next_team", "end")
        reason = result.get("reason", "LLM으로부터 이유를 받지 못했습니다.")
        feedback = result.get("feedback")

        is_backward_loop = (getattr(last_message, 'name', 'N/A') == "team2_evaluator" and next_team == "team1")
        
        if is_backward_loop:
            global_loop_count += 1
            print(f"🔄 매니저가 백워드 루프를 감지했습니다. 글로벌 루프 카운트: {global_loop_count}")
            if global_loop_count >= config.MAX_GLOBAL_LOOPS:
                print(f"❌ 글로벌 루프 제한({config.MAX_GLOBAL_LOOPS}회)을 초과하여 프로세스를 종료합니다.")
                next_team = "end" # Override the decision and force termination
                feedback = "Process terminated to prevent an infinite loop."

        print(f"🧠 매니저 결정: {next_team}, 이유: {reason}")
        
        update_dict = {
            "next_team_to_call": next_team,
            "manager_feedback": feedback,
            "global_loop_count": global_loop_count,
        }

        # 특정 팀으로 작업을 되돌려 보낼 때, 해당 팀의 재시도 횟수를 초기화
        if next_team == "team1":
            update_dict["team1_retries"] = 0
        elif next_team == "team2":
            update_dict["team2_retries"] = 0
        elif next_team == "team3":
            update_dict["team3_retries"] = 0
        
        return update_dict
    
    except Exception as e:
        print(f"❌ 매니저 에이전트 오류: {e}")
        return {"next_team_to_call": "end", "manager_feedback": "매니저 에이전트 실행 중 오류가 발생했습니다."}


def create_super_graph(team1_app, team2_app, team3_app):
    """
    매니저 에이전트를 중심으로 모든 팀 서브그래프를 통합하는 슈퍼그래프를 생성합니다.
    """
    builder = StateGraph(AgentState)

    builder.add_node("team1", team1_app)
    builder.add_node("team2", team2_app)
    builder.add_node("team3", team3_app)
    builder.add_node("manager", manager_agent)

    builder.set_entry_point("team1")

    builder.add_edge("team1", "manager")
    builder.add_edge("team2", "manager")
    builder.add_edge("team3", "manager")

    def route_from_manager(state: AgentState) -> str:
        next_team = state.get("next_team_to_call")
        print(f"🚦 슈퍼그래프 라우터: 다음 목적지는 '{next_team}'")
        if not next_team:
            return "end" # next_team이 없는 예외적인 경우 종료
        return next_team

    builder.add_conditional_edges(
        "manager",
        route_from_manager,
        {
            "team1": "team1",
            "team2": "team2",
            "team3": "team3",
            "end": END
        }
    )

    return builder.compile()