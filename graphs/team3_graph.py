# graphs/team3_graph.py

from langgraph.graph import StateGraph, END

from state import GlobalState
from agents.team3_agents import generate_answer, evaluate_answer

# 재시도 횟수 설정
MAX_RETRIES = 2

def decide_final_step(state: GlobalState) -> str:
    """
    'evaluate_answer' 노드의 결과를 바탕으로 다음 단계를 결정합니다.
    - 평가 통과 -> 서브그래프 종료 (최종 성공)
    - 평가 실패 & 재시도 횟수 남음 -> 다시 시도
    - 평가 실패 & 재시도 횟수 없음 -> 서브그래프 종료 (최종 실패)
    """
    retries = state.get("team3_retries", 0)

    if state.get("status", {}).get("team3") == "pass":
        print("✅ Team 3 통과. 최종 성공.")
        return END
    else:
        retries = state.get("team3_retries", 0)
        
        if retries <= MAX_RETRIES:
            print(f"🔁 Team 3 답변 평가 실패. 재작성을 시도합니다. ({retries}/{MAX_RETRIES})")
            return "generate_answer"
        else:
            print(f"❌ Team 3 최종 실패 (재시도 {MAX_RETRIES}회 초과).")
            state["error_message"] = "Team3: 여러 번 시도했지만 품질 기준을 충족하는 답변을 생성하지 못했습니다."
            return END

def create_team3_graph():
    """
    Team 3의 서브그래프를 생성하고 컴파일합니다.
    Workflow: Generate Answer -> Evaluate Answer -> (If fails) Retry or End
    """
    builder = StateGraph(GlobalState)

    # 1. 노드를 추가합니다.
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("evaluate_answer", evaluate_answer)

    # 2. 엣지를 연결합니다.
    builder.set_entry_point("generate_answer")
    builder.add_edge("generate_answer", "evaluate_answer")

    # 3. 조건부 엣지를 추가합니다.
    builder.add_conditional_edges(
        "evaluate_answer",
        decide_final_step,
        {
            "generate_answer": "generate_answer",
            END: END
        }
    )

    # 4. 그래프를 컴파일합니다.
    return builder.compile()