import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from agents.team2_rag_agent import agent_team2_rag_search
from agents.team2_web_agent import agent_team2_web_search
from langchain_core.runnables import RunnableLambda

PROMPT = """
[질문 요약]
{q_en_transformed}

[RAG Query]
{rag_query}

[검색된 문서]
{combined_docs}

다음 기준으로 평가하세요:
1. semantic_relevance (0~1): 질문과 의미적으로 맞는가?
2. coverage (0~1): 문서가 충분히 답을 커버하는가?
3. expertise_score = (semantic_relevance + coverage) / 2

결과를 아래 JSON으로 출력하세요:
{{
  "expertise_score": float
}}
"""

def evaluate_docs(docs, q_summary, query, llm, chain):
    if not docs:
        return 0.0
    from langchain.schema import Document
    filtered_docs = []
    for doc in docs:
        if hasattr(doc, "page_content"):
            filtered_docs.append(doc)
        elif isinstance(doc, str) and doc.strip():
            filtered_docs.append(Document(page_content=doc))
    if not filtered_docs:
        return 0.0
    combined_docs = "\n\n".join([doc.page_content for doc in filtered_docs])
    try:
        result = chain.invoke({
            "q_en_transformed": q_summary,
            "rag_query": query,
            "combined_docs": combined_docs
        })
        parsed = json.loads(result.content)
        return parsed.get("expertise_score", 0.0)
    except Exception as e:
        print(f"❌ 평가 오류: {e}")
        return 0.0

def supervisor_team2(state: dict) -> dict:
    print(f"🔍 Team 2 Supervisor 시작")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate.from_template(PROMPT)
    chain = prompt | llm

    q_summary = state.get("q_en_transformed", "")
    query = state.get("rag_query", "")

    max_attempts = 2

    # === RAG 탐색 최대 2회 ===
    for attempt in range(max_attempts):
        print(f"🔎 RAG 문서 검색 ({attempt+1}차)")
        state = RunnableLambda(agent_team2_rag_search).invoke(state)
        docs = state.get("rag_docs", [])
        score = evaluate_docs(docs, q_summary, query, llm, chain)
        if score >= 0.5:
            print(f"✅ Team 2 통과 (RAG {attempt+1}차, score={score:.2f})")
            state["status"]["team2"] = "pass"
            return state
        else:
            print(f"🔁 RAG {attempt+1}차 실패 (score={score:.2f})")

    # === WEB 탐색 최대 2회 ===
    for attempt in range(max_attempts):
        print(f"🌐 WEB 문서 검색 ({attempt+1}차)")
        state = RunnableLambda(agent_team2_web_search).invoke(state)
        docs = state.get("web_docs", [])
        score = evaluate_docs(docs, q_summary, query, llm, chain)
        if score >= 0.5:
            print(f"✅ Team 2 통과 (WEB {attempt+1}차, score={score:.2f})")
            state["status"]["team2"] = "pass"
            return state
        else:
            print(f"🔁 WEB {attempt+1}차 실패 (score={score:.2f})")

    # === 모든 시도 실패 ===
    print("❌ Team 2 최종 실패")
    state["status"]["team2"] = "fail"
    return state
