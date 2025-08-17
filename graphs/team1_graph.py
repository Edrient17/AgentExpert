# graphs/team1_graph.py

from langgraph.graph import StateGraph, END

from state import GlobalState
from agents.team1_agents import process_question, evaluate_question

# 재시도 횟수 설정
MAX_RETRIES = 2

def decide_next_step(state: GlobalState) -> str:
    """
    'evaluate_question' 노드의 결과를 바탕으로 다음 단계를 결정합니다.
    - 평가 통과 -> 서브그래프 종료
    - 평가 실패 & 재시도 횟수 남음 -> 다시 시도
    - 평가 실패 & 재시도 횟수 없음 -> 서브그래프 종료 (실패 상태)
    """
    # 상태에서 재시도 횟수를 가져오거나 초기화합니다.
    retries = state.get("team1_retries", 0)
    
    # 평가 결과를 가져옵니다.
    if state.get("status", {}).get("team1") == "pass":
        print("✅ Team 1 통과.")
        state["team1_retries"] = 0
        return END
    else:
        retries = state.get("team1_retries", 0)
        
        if retries < MAX_RETRIES:
            print(f"🔁 Team 1 실패. 재시도합니다. ({retries}/{MAX_RETRIES})")
            return "process_question"
        else:
            print(f"❌ Team 1 최종 실패 (재시도 {MAX_RETRIES}회 초과).")
            return END

def create_team1_graph():
    """
    Team 1의 서브그래프를 생성하고 컴파일합니다.
    """
    builder = StateGraph(GlobalState)

    # 1. 노드를 추가합니다.
    builder.add_node("process_question", process_question)
    builder.add_node("evaluate_question", evaluate_question)

    # 2. 엣지를 연결합니다.
    builder.set_entry_point("process_question")
    builder.add_edge("process_question", "evaluate_question")
    
    # 3. 조건부 엣지를 추가합니다.
    # 'evaluate_question' 노드가 끝난 후, 'decide_next_step' 함수의 결과에 따라 분기합니다.
    builder.add_conditional_edges(
        "evaluate_question",
        decide_next_step,
        {
            "process_question": "process_question",
            END: END
        }
    )

    # 4. 그래프를 컴파일합니다.
    return builder.compile()