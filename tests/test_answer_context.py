from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, ToolMessage

from src.agents.answer.context import get_answer_context
from src.schema.constants import NodeName, WorkflowSignal


def test_answer_context_uses_original_user_input_and_team1_metadata():
    state = {
        "messages": [
            HumanMessage(content="Original question"),
            ToolMessage(
                content=WorkflowSignal.PASS,
                name=NodeName.QUERY_EVALUATOR,
                tool_call_id="test-team1",
                additional_kwargs={
                    "q_en_transformed": "Refined question",
                    "output_format": ["qa", "en"],
                },
            ),
        ]
    }

    context = get_answer_context(state)

    assert context["user_input"] == "Original question"
    assert context["q_en_transformed"] == "Refined question"
    assert context["output_format"] == ["qa", "en"]


def test_answer_context_prefers_state_docs_with_rag_first():
    rag_doc = Document(page_content="rag")
    web_doc = Document(page_content="web")
    state = {
        "messages": [HumanMessage(content="Question")],
        "rag_docs": [rag_doc],
        "web_docs": [web_doc],
    }

    context = get_answer_context(state)

    assert context["docs"] == [rag_doc, web_doc]
    assert context["rag_docs"] == [rag_doc]
    assert context["web_docs"] == [web_doc]
