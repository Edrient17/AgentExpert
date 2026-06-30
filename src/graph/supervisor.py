from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

import config
from src.prompts.loader import load_prompt_template
from src.schema.constants import NodeName, SimpleQuery, Team, WorkflowSignal
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
        if last_name == NodeName.QUERY_EVALUATOR and str(last_content).strip() == WorkflowSignal.PASS:
            is_simple = state.get("is_simple_query", SimpleQuery.NO)
            if is_simple == SimpleQuery.YES:
                print("[route] Supervisor shortcut: simple query -> Answer Team")
                return {
                    "next_team_to_call": Team.ANSWER,
                    "manager_feedback": None,
                    "global_loop_count": global_loop_count,
                    "team3_retries": 0,
                }
            print("[route] Supervisor shortcut: retrieval query -> Retrieval Team")
            return {
                "next_team_to_call": Team.RETRIEVAL,
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
        next_team = result.get("next_team", Team.END)
        reason = result.get("reason", "No reason was returned by the LLM.")
        feedback = result.get("feedback")

        is_t1_loop = last_name == NodeName.QUERY_EVALUATOR and next_team == Team.QUERY
        is_t2_loop = last_name == NodeName.RETRIEVAL_EVALUATOR and next_team == Team.QUERY
        is_t3_loop = last_name == NodeName.ANSWER_EVALUATOR and next_team == Team.ANSWER

        if is_t1_loop or is_t2_loop or is_t3_loop:
            global_loop_count += 1
            print(f"[loop] Supervisor detected a backward loop. Global loop count: {global_loop_count}")
            if global_loop_count >= config.MAX_GLOBAL_LOOPS:
                print(f"[error] Global loop limit exceeded ({config.MAX_GLOBAL_LOOPS}). Ending workflow.")
                error_content = "The internal processing limit was exceeded, so an answer could not be generated. Please revise the question and try again."
                return {
                    "next_team_to_call": Team.END,
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

        if next_team == Team.QUERY:
            update_dict["team1_retries"] = 0
        elif next_team == Team.RETRIEVAL:
            update_dict["team2_retries"] = 0
        elif next_team == Team.ANSWER:
            update_dict["team3_retries"] = 0

        return update_dict
    except Exception as e:
        print(f"[error] Supervisor error: {e}")
        return {"next_team_to_call": Team.END, "manager_feedback": "An error occurred while running the supervisor agent."}


def create_supervisor_graph(query_team_app, retrieval_team_app, answer_team_app):
    builder = StateGraph(AgentState)

    builder.add_node(Team.QUERY, query_team_app)
    builder.add_node(Team.RETRIEVAL, retrieval_team_app)
    builder.add_node(Team.ANSWER, answer_team_app)
    builder.add_node("manager", manager_agent)

    builder.set_entry_point(Team.QUERY)

    builder.add_edge(Team.QUERY, "manager")
    builder.add_edge(Team.RETRIEVAL, "manager")
    builder.add_edge(Team.ANSWER, "manager")

    def route_from_manager(state: AgentState) -> str:
        next_team = state.get("next_team_to_call")
        print(f"[route] Supergraph router: next destination is '{next_team}'")
        if not next_team:
            return Team.END
        return next_team

    builder.add_conditional_edges(
        "manager",
        route_from_manager,
        {
            Team.QUERY: Team.QUERY,
            Team.RETRIEVAL: Team.RETRIEVAL,
            Team.ANSWER: Team.ANSWER,
            Team.END: END,
        },
    )

    return builder.compile()
