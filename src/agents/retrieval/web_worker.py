import uuid
from typing import Any, Dict

from langchain_core.messages import ToolMessage

from src.schema.state import AgentState
from src.tools.common import format_docs
from src.tools.web_research import deep_research_web_search


WEB_SEARCH_NUM = 5


def web_search(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Retrieval Web Worker (web search) ---")
    q_en_transformed = state.get("q_en_transformed", "")
    try:
        web_docs = deep_research_web_search.func(q_en_transformed, max_results=WEB_SEARCH_NUM)
        return {
            "messages": [
                ToolMessage(
                    content=format_docs(web_docs),
                    name="web_search_result",
                    tool_call_id=str(uuid.uuid4()),
                    additional_kwargs={"source_docs": web_docs},
                )
            ]
        }
    except Exception as e:
        print(f"[error] Retrieval Web Worker tool error: {e}")
        return {
            "messages": [
                ToolMessage(
                    content=f"fail: Web search error - {e}",
                    name="web_search",
                    tool_call_id=str(uuid.uuid4()),
                )
            ]
        }
