# team3_answer_agent.py

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

ANSWER_PROMPT = """
당신은 전문적인 기술 문서를 바탕으로 질문에 정확하고 자세하게 답변해야 합니다.

아래의 문서 내용을 참고하여, 주어진 질문 요약과 RAG 쿼리를 바탕으로 가능한 한 구체적이고 신뢰도 높은 답변을 생성하세요.
문서에 포함되지 않은 정보는 추론하거나 상상하지 말고, "문서에 해당 정보가 없습니다"라고 하세요.
답변은 반드시 한국어로 작성해야 합니다.

[질문 요약]
{q_en_transformed}

[RAG 쿼리]
{rag_query}

[참고 문서]
{combined_docs}

답변:
"""

def agent_team3_answer_generation(state: dict) -> dict:
    question = state.get("q_en_transformed", "")
    query = state.get("rag_query", "")

    web_docs = state.get("web_docs", [])
    rag_docs = state.get("rag_docs", [])

    if web_docs:  # web_docs가 비어있지 않으면 web_docs만 사용
        docs = web_docs
    else:         # web_docs가 비어 있으면 rag_docs만 사용
        docs = rag_docs

    combined_docs = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatOpenAI(model="gpt-4")
    prompt = PromptTemplate.from_template(ANSWER_PROMPT)
    chain = prompt | llm

    try:
        result = chain.invoke({
            "q_en_transformed": question,
            "rag_query": query,
            "combined_docs": combined_docs
        })

        state["generated_answer"] = result.content.strip()
        return state

    except Exception as e:
        print(f"❌ Team 3 Agent 오류: {e}")
        state["generated_answer"] = "답변 생성 실패"
        return state
