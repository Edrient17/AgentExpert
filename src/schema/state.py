from __future__ import annotations

from operator import add as list_concat
from typing import Annotated, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.schema.constants import SimpleQueryValue, TeamName


class AgentState(TypedDict, total=False):
    messages: Annotated[List[AnyMessage], add_messages]

    next_team_to_call: Optional[TeamName]
    manager_feedback: Optional[str]

    user_input: Optional[str]
    q_en_transformed: Optional[str]
    best_rag_query: Optional[str]

    rag_docs: Annotated[List[Document], list_concat]
    web_docs: Annotated[List[Document], list_concat]

    is_simple_query: SimpleQueryValue

    team1_retries: int
    team2_retries: int
    team3_retries: int
    global_loop_count: int
