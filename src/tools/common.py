from typing import List

from langchain_core.documents import Document


def format_docs(docs: List[Document], max_chars: int = 15000) -> str:
    contents = [d.page_content.strip() for d in docs if isinstance(d, Document) and d.page_content]
    if not contents:
        return "[NO CONTENT]"
    joined = "\n\n---\n\n".join(contents)
    if len(joined) > max_chars:
        return joined[:max_chars] + "\n\n...[content truncated]..."
    return joined
