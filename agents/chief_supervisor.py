# agents/chief_supervisor.py
import json
from typing import Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from utils import get_llm

class RoutingDecision(BaseModel):
    next_node: str = Field(description="The name of the node to move to. It must be one of 'team1_supervisor', 'team2_supervisor', 'team3_supervisor', or 'end'.")
    reasoning: str = Field(description="A brief explanation of why this node was selected.")

parser = JsonOutputParser(pydantic_object=RoutingDecision)

ROUTING_PROMPT = PromptTemplate.from_template("""
You are the Chief Supervisor, a master router in a multi-agent system.
Based on the current state of the project, decide which supervisor to call next.

Available nodes:
- team1_supervisor: Analyzes and reformulates the user's question. Call this first.
- team2_supervisor: Retrieves documents from a vector store or the web. Call this if the question requires external knowledge.
- team3_supervisor: Generates the final answer using the provided context. Call this when enough information is gathered or no retrieval is needed.
- end: The process is complete or has failed irrecoverably.

Current State:
{state_json}

Decision process (evaluate in this specific order):
1.  **Termination Check**: If 'status' shows 'team3' as 'pass', the work is complete. -> Go to 'end'.
2.  **Failure Check**: If 'status' shows any team as 'fail'. -> Go to 'end'.
3.  **Answer Generation**: If 'status' shows 'team1' and 'team2' as 'pass' but not 'team3'. -> Go to 'team3_supervisor'.
4.  **Retrieval or Skip**: If 'status' shows 'team1' as 'pass' but not 'team2'. Analyze the question in the state.
    - If the question requires external documents/knowledge (e.g., "Tell me about LangGraph"). -> Go to 'team2_supervisor'.
    - If the question does NOT require external knowledge (e.g., "Hi? What's your name?", "What is 2+2?"). -> Go to 'team3_supervisor'.
5.  **Default/Start**: Otherwise (e.g., 'status' is empty), start the process. -> Go to 'team1_supervisor'.


Based on the logic above, what is the next node?

{schema}
""").partial(schema=parser.get_format_instructions())


def chief_supervisor_router(state: Dict) -> Dict:

    print("🤖 Chief Supervisor decides what to do...")
    
    print(f"Current Status: {state.get('status')}")
    print(f"Incoming next_node: {state.get('next_node')}")

    state_for_llm = {
        "user_input": state.get("user_input"),
        "q_en_transformed": state.get("q_en_transformed"),
        "rag_query": state.get("rag_query"),
        "rag_docs_count": len(state.get("rag_docs", [])),
        "web_docs_count": len(state.get("web_docs", [])),
        "generated_answer": bool(state.get("generated_answer")),
        "status": state.get("status"),
        "error_message": state.get("error_message")
    }
    state_json = json.dumps(state_for_llm, ensure_ascii=False, indent=2)

    llm = get_llm(temperature=0)
    chain = ROUTING_PROMPT | llm | parser

    try:
        decision = chain.invoke({"state_json": state_json})
        next_node = decision.get("next_node", "end")
        print(f"🧠 라우팅 결정: {next_node} (이유: {decision.get('reasoning')})")
        state["next_node"] = next_node
    except Exception as e:
        print(f"❌ Chief Supervisor 라우팅 실패: {e}")
        state["next_node"] = "end"

    return state