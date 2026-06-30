from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
from src.prompts.loader import load_prompt_template


class SearchResult(BaseModel):
    title: str = Field(description="The title of the search result.")
    url: str = Field(description="The URL of the search result.")
    summary: str = Field(description="A detailed, objective summary addressing the user's query.")


class SearchResults(BaseModel):
    results: List[SearchResult]


@tool
def deep_research_web_search(query: str, max_results: int = 3) -> List[Document]:
    """
    Generate structured web research summaries using an LLM.
    """
    print(f"[tool] Running Deep Research tool: '{query}'")
    if not query:
        return []

    try:
        llm = ChatOpenAI(model=getattr(config, "LLM_MODEL_WEB", "gpt-4.1"), temperature=0)
        structured_llm = llm.with_structured_output(SearchResults)
        prompt = load_prompt_template("retrieval/web_research.yaml")
        chain = prompt | structured_llm
        response: SearchResults = chain.invoke({"query": query, "max_results": max_results})

        docs: List[Document] = []
        if response and response.results:
            for r in response.results:
                docs.append(
                    Document(
                        page_content=f"Title: {r.title}\nSummary: {r.summary}",
                        metadata={"source": r.url, "title": r.title},
                    )
                )
        return docs
    except Exception as e:
        print(f"[error] Deep Research tool error: {e}")
        return []
