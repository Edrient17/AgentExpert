from typing import Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

import config
from src.prompts.loader import load_chat_prompt_template


class SimpleQueryResult(BaseModel):
    is_simple_query: Literal["Yes", "No"]


@tool
def classify_simple_query(user_question: str) -> dict:
    """
    Classify whether the question can skip retrieval.
    """
    llm = ChatOpenAI(model=getattr(config, "LLM_MODEL_TEAM1", "gpt-4.1"), temperature=0)
    prompt = load_chat_prompt_template("query/simple_query_classifier.yaml")
    chain = prompt | llm.with_structured_output(SimpleQueryResult)

    try:
        result = chain.invoke({"question": user_question})
        return result.is_simple_query
    except Exception as e:
        print(f"[warn] classify_simple_query failed: {e}")
        return "No"
