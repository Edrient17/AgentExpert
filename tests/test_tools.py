from langchain_core.documents import Document

from src.tools.common import format_docs


def test_format_docs_returns_no_content_for_empty_input():
    assert format_docs([]) == "[NO CONTENT]"


def test_format_docs_joins_document_content():
    docs = [Document(page_content=" first "), Document(page_content="second")]
    assert format_docs(docs) == "first\n\n---\n\nsecond"


def test_format_docs_truncates_long_content():
    docs = [Document(page_content="abcdef")]
    assert format_docs(docs, max_chars=3).startswith("abc")
    assert "[content truncated]" in format_docs(docs, max_chars=3)
