from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

import config
from src.prompts.loader import load_prompt_template
from src.schema.outputs import ManagerDecision
from src.schema.state import AgentState


def manager_agent(state: AgentState) -> dict:
    print("--- SUPERVISOR: Review work and decide next step ---")

    global_loop_count = state.get("global_loop_count", 0)
    last_message = state["messages"][-1]
    last_name = getattr(last_message, "name", "N/A")
    last_content = getattr(last_message, "content", "")
    user_question = next((msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)), "")

    try:
        if last_name == "team1_evaluator" and str(last_content).strip() == "pass":
            is_simple = state.get("is_simple_query", "No")
            if is_simple == "Yes":
                print("[route] Supervisor shortcut: simple query -> Answer Team")
                return {
                    "next_team_to_call": "team3",
                    "manager_feedback": None,
                    "global_loop_count": global_loop_count,
                    "team3_retries": 0,
                }
            print("[route] Supervisor shortcut: retrieval query -> Retrieval Team")
            return {
                "next_team_to_call": "team2",
                "manager_feedback": None,
                "global_loop_count": global_loop_count,
                "team2_retries": 0,
            }
    except Exception as e:
        print(f"[warn] is_simple_query routing failed: {e}. Falling back to LLM routing.")

    parser = JsonOutputParser(p_object=ManagerDecision)
    prompt = load_prompt_template("supervisor/manager.yaml")
    llm = ChatOpenAI(
        model=config.LLM_MODEL_SUPER_ROUTER,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    chain = prompt.partial(schema=parser.get_format_instructions()) | llm | parser

    try:
        result = chain.invoke(
            {
                "user_question": user_question,
                "last_message_name": last_name,
                "last_message_content": last_content,
            }
        )
        next_team = result.get("next_team", "end")
        reason = result.get("reason", "No reason was returned by the LLM.")
        feedback = result.get("feedback")

        is_t1_loop = last_name == "team1_evaluator" and next_team == "team1"
        is_t2_loop = last_name == "team2_evaluator" and next_team == "team1"
        is_t3_loop = last_name == "team3_evaluator" and next_team == "team3"

        if is_t1_loop or is_t2_loop or is_t3_loop:
            global_loop_count += 1
            print(f"[loop] Supervisor detected a backward loop. Global loop count: {global_loop_count}")
            if global_loop_count >= config.MAX_GLOBAL_LOOPS:
                print(f"[error] Global loop limit exceeded ({config.MAX_GLOBAL_LOOPS}). Ending workflow.")
                error_content = "The internal processing limit was exceeded, so an answer could not be generated. Please revise the question and try again."
                return {
                    "next_team_to_call": "end",
                    "manager_feedback": "Process terminated to prevent an infinite loop.",
                    "global_loop_count": global_loop_count,
                    "messages": [AIMessage(content=error_content)],
                }

        print(f"[decision] Supervisor decision: {next_team}, reason: {reason}")

        update_dict = {
            "next_team_to_call": next_team,
            "manager_feedback": feedback,
            "global_loop_count": global_loop_count,
        }

        if next_team == "team1":
            update_dict["team1_retries"] = 0
        elif next_team == "team2":
            update_dict["team2_retries"] = 0
        elif next_team == "team3":
            update_dict["team3_retries"] = 0

        return update_dict
    except Exception as e:
        print(f"[error] Supervisor error: {e}")
        return {"next_team_to_call": "end", "manager_feedback": "An error occurred while running the supervisor agent."}


def create_supervisor_graph(query_team_app, retrieval_team_app, answer_team_app):
    builder = StateGraph(AgentState)

    builder.add_node("team1", query_team_app)
    builder.add_node("team2", retrieval_team_app)
    builder.add_node("team3", answer_team_app)
    builder.add_node("manager", manager_agent)

    builder.set_entry_point("team1")

    builder.add_edge("team1", "manager")
    builder.add_edge("team2", "manager")
    builder.add_edge("team3", "manager")

    def route_from_manager(state: AgentState) -> str:
        next_team = state.get("next_team_to_call")
        print(f"[route] Supergraph router: next destination is '{next_team}'")
        if not next_team:
            return "end"
        return next_team

    builder.add_conditional_edges(
        "manager",
        route_from_manager,
        {
            "team1": "team1",
            "team2": "team2",
            "team3": "team3",
            "end": END,
        },
    )

    return builder.compile()
