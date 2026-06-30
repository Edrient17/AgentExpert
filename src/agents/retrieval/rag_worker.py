import uuid
from typing import Any, Dict

from langchain_core.messages import ToolMessage

from src.agents.retrieval.context import get_query_from_history
from src.schema.constants import NodeName, fail_signal
from src.schema.state import AgentState
from src.tools.common import format_docs
from src.tools.rag import vector_store_rag_search


RAG_SEARCH_NUM = 7


def rag_search(state: AgentState) -> Dict[str, Any]:
    print("--- AGENT: Retrieval RAG Worker (RAG search) ---")
    rag_query = get_query_from_history(state)
    if not rag_query:
        return {
            "messages": [
                ToolMessage(
                    content=fail_signal("RAG query was not found."),
                    name=NodeName.RAG_SEARCH,
                    tool_call_id=str(uuid.uuid4()),
                )
            ]
        }

    try:
        rag_docs = vector_store_rag_search.func(rag_query, top_k=RAG_SEARCH_NUM, rerank_k=RAG_SEARCH_NUM)
        return {
            "messages": [
                ToolMessage(
                    content=format_docs(rag_docs),
                    name=NodeName.RAG_SEARCH_RESULT,
                    tool_call_id=str(uuid.uuid4()),
                    additional_kwargs={"source_docs": rag_docs},
                )
            ],
            "rag_docs": [],
            "web_docs": [],
        }
    except Exception as e:
        print(f"[error] Retrieval RAG Worker tool error: {e}")
        return {
            "messages": [
                ToolMessage(
                    content=fail_signal(f"RAG search error - {e}"),
                    name=NodeName.RAG_SEARCH,
                    tool_call_id=str(uuid.uuid4()),
                )
            ]
        }
