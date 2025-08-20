# graphs/team1_graph.py

from langgraph.graph import StateGraph, END
from state import AgentState
from agents.team1_agents import process_question, evaluate_question

def create_team1_graph():
    """
    Team 1의 서브그래프를 생성하고 컴파일합니다.
    """
    builder = StateGraph(AgentState)

    builder.add_node("process_question", process_question)
    builder.add_node("evaluate_question", evaluate_question)

    builder.set_entry_point("process_question")
    builder.add_edge("process_question", "evaluate_question")
    
    def route_after_evaluation(state: AgentState) -> str:
        last_message = state['messages'][-1]
        
        if last_message.content == "retry":
            print("🚦 라우터: Team 1 재시도 결정.")
            return "process_question"
        else:
            print("🚦 라우터: Team 1 종료 결정.")
            return END

    builder.add_conditional_edges(
        "evaluate_question",
        route_after_evaluation,
        {
            "process_question": "process_question",
            END: END
        }
    )

    return builder.compile()