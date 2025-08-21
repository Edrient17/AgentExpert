# graphs/team3_graph.py

from langgraph.graph import StateGraph, END
from state import AgentState
from agents.team3_agents import generate_answer, evaluate_answer

def create_team3_graph():
    """
    Team 3의 서브그래프를 생성하고 컴파일합니다.
    """
    builder = StateGraph(AgentState)

    builder.add_node("generate_answer", generate_answer)
    builder.add_node("evaluate_answer", evaluate_answer)

    builder.set_entry_point("generate_answer")
    builder.add_edge("generate_answer", "evaluate_answer")

    def route_after_evaluation(state: AgentState) -> str:
        """
        evaluate_answer 노드가 보낸 메시지를 확인하여 다음 단계를 결정합니다.
        - "pass" 또는 "fail" -> 서브그래프 종료
        - "retry" -> generate_answer 노드로 돌아가 재시도
        """
        last_message = state['messages'][-1]
        
        if last_message.content.startswith("retry"):
            print("🚦 라우터: Team 3 재시도 결정.")
            return "generate_answer"
        else: # "pass" or "fail"
            print("🚦 라우터: Team 3 종료 결정.")
            return END

    builder.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluation,
        {
            "generate_answer": "generate_answer",
            END: END
        }
    )

    return builder.compile()
