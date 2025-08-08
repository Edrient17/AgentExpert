import os
import requests
from langchain.schema import Document

def agent_team2_web_search(state: dict, max_results=5) -> dict:
    query = state.get("rag_query", "").strip()
    new_state = state.copy()

    if not query:
        print("⚠️ rag_query 없음")
        new_state["web_docs"] = []
        return new_state

    api_key = os.getenv("SERPAPI_API_KEY")  # 환경변수에 SerpAPI 키 저장
    endpoint = "https://serpapi.com/search"
    params = {
        "engine": "google_light",
        "q": query,
        "api_key": api_key
    }

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        docs = []

        # 1. Answer box 우선 저장 (예: 'answer_box', 'knowledge_graph' 등)
        answer_box = data.get("answer_box")
        if answer_box:
            answer = answer_box.get("answer") or answer_box.get("snippet") or answer_box.get("highlighted") or ""
            if answer:
                docs.append(Document(
                    page_content=answer.strip(),
                    metadata={"source": "answer_box", "title": answer_box.get("title", "")}
                ))

        # 2. Knowledge Graph
        knowledge_graph = data.get("knowledge_graph")
        if knowledge_graph:
            desc = knowledge_graph.get("description") or ""
            if desc:
                docs.append(Document(
                    page_content=desc.strip(),
                    metadata={"source": "knowledge_graph", "title": knowledge_graph.get("title", "")}
                ))

        # 3. Organic Results
        organic_results = data.get("organic_results", [])
        for item in organic_results[:max_results - len(docs)]:
            snippet = item.get("snippet", "")
            if snippet:
                docs.append(Document(
                    page_content=snippet.strip(),
                    metadata={"source": item.get("link", ""), "title": item.get("title", "")}
                ))

        new_state["web_docs"] = docs
        # docs 출력
        if docs:
            print(f"🔍 {len(docs)}개의 웹 문서 검색 결과:")
            for doc in docs:
                print(f"- {doc.metadata.get('title', '제목 없음')}: {doc.page_content[:100]}...")  # 내용 일부 출력
        else:
            print("⚠️ 검색 결과가 없습니다.")
        return new_state

    except Exception as e:
        print(f"❌ SerpAPI 검색 실패: {e}")
        new_state["web_docs"] = []
        return new_state
