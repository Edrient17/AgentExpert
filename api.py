import uuid
import os

from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from mcp.server.fastmcp import FastMCP

from graph_factory import get_graph_app
from state import AgentState

# --- Create MCP Server ---
# Create a server that can communicate with Claude using FastMCP.
mcp = FastMCP(
    name="My Professional RAG Agent",
    instructions="A multi-agent system made with LangGraph. It takes questions and generates professional answers.",
)

# Load LangGraph
langgraph_app = get_graph_app()

class Answer(BaseModel):
    final_answer: str
    answer_generation_successful: bool

@mcp.tool()
def ask_agent(question: str) -> Answer:
    """
    After taking questions and running the multi-agent RAG pipeline,
    return final_answer and answer_generation_successful.
    """
    print(f"--- MCP Tool 'ask_agent' 실행 ---")
    print(f"Input Questions: {question}")
    try:
        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "team1_retries": 0, "team2_retries": 0, "team3_retries": 0,
            "global_loop_count": 0, "is_simple_query": "No"
        }
        thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

        final_state = langgraph_app.invoke(initial_state, thread)

        final_answer = "Failed to generate answer."
        if messages := final_state.get("messages"):
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    final_answer = msg.content
                    break

        print(f"Generated Answer: {final_answer[:100]}...")
        return Answer(final_answer=final_answer, answer_generation_successful=True)

    except Exception as e:
        print(f"Error occured while running MCP Tool: {e}")
        return Answer(final_answer=f"An error occurred and the answer could not be generated: {e}", answer_generation_successful=False)

# --- Run Server ---
if __name__ == "__main__":
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = int(os.environ.get("PORT", "8080"))
    mcp.run(transport="streamable-http")