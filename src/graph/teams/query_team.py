from langgraph.graph import END, StateGraph

from src.agents.query.evaluator import evaluate_question
from src.agents.query.worker import process_question
from src.schema.constants import is_retry_signal
from src.schema.state import AgentState


def create_query_team_graph():
    builder = StateGraph(AgentState)

    builder.add_node("process_question", process_question)
    builder.add_node("evaluate_question", evaluate_question)

    builder.set_entry_point("process_question")
    builder.add_edge("process_question", "evaluate_question")

    def route_after_evaluation(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if is_retry_signal(last_message.content):
            print("[route] Query Team retry.")
            return "process_question"
        print("[route] Query Team finished.")
        return END

    builder.add_conditional_edges(
        "evaluate_question",
        route_after_evaluation,
        {
            "process_question": "process_question",
            END: END,
        },
    )

    return builder.compile()
