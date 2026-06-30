from langgraph.graph import END, StateGraph

from src.agents.answer.evaluator import evaluate_answer
from src.agents.answer.generator import generate_answer
from src.schema.constants import is_retry_signal
from src.schema.state import AgentState


def create_answer_team_graph():
    builder = StateGraph(AgentState)

    builder.add_node("generate_answer", generate_answer)
    builder.add_node("evaluate_answer", evaluate_answer)

    builder.set_entry_point("generate_answer")
    builder.add_edge("generate_answer", "evaluate_answer")

    def route_after_evaluation(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if is_retry_signal(last_message.content):
            print("[route] Answer Team retry.")
            return "generate_answer"
        print("[route] Answer Team finished.")
        return END

    builder.add_conditional_edges(
        "evaluate_answer",
        route_after_evaluation,
        {
            "generate_answer": "generate_answer",
            END: END,
        },
    )

    return builder.compile()
