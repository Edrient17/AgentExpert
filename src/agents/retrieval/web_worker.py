import uuid
from typing import Any, Dict

from langchain_core.messages import ToolMessage

import config
from src.schema.constants import NodeName, fail_signal
from src.schema.state import AgentState
from src.tools.common import format_docs
from src.tools.web_research import deep_research_web_search


WEB_SEARCH_NUM = 5


def web_search(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Retrieval Web Worker (web search) ---")
    q_en_transformed = state.get("q_en_transformed", "")
    if not config.ENABLE_WEB_RESEARCH:
        return {
            "messages": [
                ToolMessage(
                    content=fail_signal("Web research is disabled by configuration."),
                    name=NodeName.WEB_SEARCH,
                    tool_call_id=str(uuid.uuid4()),
                )
            ]
        }
    try:
        web_docs = deep_research_web_search.func(q_en_transformed, max_results=WEB_SEARCH_NUM)
        return {
            "messages": [
                ToolMessage(
                    content=format_docs(web_docs),
                    name=NodeName.WEB_SEARCH_RESULT,
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
                    content=fail_signal(f"Web search error - {e}"),
                    name=NodeName.WEB_SEARCH,
                    tool_call_id=str(uuid.uuid4()),
                )
            ]
        }
