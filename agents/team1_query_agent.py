# team1_agent.py

import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Agent 1: 사용자 질문 → 유효성 판단 + 영어 번역 + RAG query 생성
PROMPT_TEMPLATE = """
다음 사용자의 질문을 분석해서 아래 세 가지 항목을 생성하세요:

1. q_validity: 이 입력이 질문으로서 유효한지 (true/false)
2. q_en_transformed: 질문을 영어로 자연스럽게 변환한 것
3. rag_query: RAG 검색을 위한 간결하고 핵심적인 형태의 영어 쿼리

다음 JSON 형식으로 출력하세요:

{{
  "q_validity": true or false,
  "q_en_transformed": "...",
  "rag_query": "..."
}}

[사용자 질문]
{user_input}
"""

def agent_team1_question_processing(state: dict) -> dict:
    """
    Team 1 Agent:
    - user_input을 받아
    - q_validity, q_en_transformed, rag_query를 생성
    - state를 업데이트하여 반환
    """
    user_input = state.get("user_input", "")
    new_state = state.copy()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm

    try:
        result = chain.invoke({"user_input": user_input})
        parsed = json.loads(result.content)

        new_state["q_validity"] = parsed.get("q_validity", False)
        new_state["q_en_transformed"] = parsed.get("q_en_transformed", "")
        new_state["rag_query"] = parsed.get("rag_query", "")

    except Exception as e:
        print(f"❌ Team 1 Agent 오류: {e}")
        new_state["q_validity"] = False
        new_state["q_en_transformed"] = ""
        new_state["rag_query"] = ""

    return new_state
