from langchain_core.messages import ToolMessage

import config
from src.agents.retrieval.evaluator import evaluate_documents
from src.agents.retrieval.web_worker import web_search
from src.schema.constants import NodeName, WorkflowSignal


def test_retrieval_evaluator_fails_instead_of_web_fallback_when_web_disabled(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_WEB_RESEARCH", False)
    state = {
        "messages": [
            ToolMessage(
                content="[NO CONTENT]",
                name=NodeName.RAG_SEARCH_RESULT,
                tool_call_id="rag",
                additional_kwargs={"source_docs": []},
            )
        ],
        "rag_docs": [],
        "web_docs": [],
        "team2_retries": 0,
    }

    result = evaluate_documents(state)
    message = result["messages"][0]

    assert message.name == NodeName.RETRIEVAL_EVALUATOR
    assert message.content == WorkflowSignal.FAIL
    assert message.additional_kwargs["failed_reason"] == "web_research_disabled"


def test_web_worker_returns_failure_when_web_disabled(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_WEB_RESEARCH", False)
    result = web_search({"q_en_transformed": "What is AgentExpert?"})
    message = result["messages"][0]

    assert message.name == NodeName.WEB_SEARCH
    assert message.content.startswith(WorkflowSignal.FAIL)
