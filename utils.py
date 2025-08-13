# utils.py

import os
from typing import List, Union
from langchain_openai import ChatOpenAI
from langchain.schema import Document

# LLM 클라이언트 생성 함수
def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0) -> ChatOpenAI:
    """
    중앙에서 관리되는 LLM 클라이언트 생성 함수.
    API 키가 설정되지 않았을 경우 RuntimeError를 발생시킴.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Check .env or environment variables.")
    return ChatOpenAI(model=model_name, temperature=temperature)

# 문서 포맷팅 함수
def format_docs_for_prompt(docs: List[Union[str, Document]], max_chars: int = 10000) -> str:
    """
    LangChain Document 객체 또는 문자열 리스트를 받아
    LLM 프롬프트에 삽입하기 좋은 단일 문자열로 결합.
    """
    contents: List[str] = []
    for d in docs:
        try:
            text = getattr(d, "page_content", None)
            if text is None and isinstance(d, str):
                text = d
            
            if text:
                stripped_text = str(text).strip()
                if stripped_text:
                    contents.append(stripped_text)
        except Exception:
            continue
    
    joined = "\n\n---\n\n".join(contents)
    
    if not joined:
        return "[NO CONTENT]"
    
    if len(joined) > max_chars:
        return joined[:max_chars] + "\n\n...[truncated]..."
        
    return joined