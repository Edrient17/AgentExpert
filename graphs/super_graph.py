# graphs/super_graph.py

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

import config
from state import GlobalState

def retrieval_necessity_router(state: GlobalState) -> str:
    """
    Team 1 완료 후, LLM을 이용해 Team 2(정보 검색)가 필요한지 판단하는 **라우팅 함수**입니다.
    """
    print("--- ROUTER: 정보 검색 필요 여부 판단 ---")
    
    if state.get("status", {}).get("team1") == "fail":
        print("🚦 라우터: Team 1 실패 감지. 워크플로우를 종료합니다.")
        return END

    question = state.get("q_en_transformed", "")
    
    prompt = PromptTemplate.from_template("""
You are a meticulous and safety-conscious router in a Q&A pipeline. Your critical task is to determine if a user's question can be answered *reliably* with your general knowledge, or if it requires consulting specific, external information to ensure accuracy and currency.

Your response must be a single word: either 'retrieve' or 'skip'.

**Decision Criteria:**

1.  **retrieve**: Choose this if the question involves any of the following:
    * **Specific Products/Services:** Mentions specific models, brands, software versions (e.g., "ASUS Z790 motherboard", "Photoshop 2024", "LangGraph v0.1").
    * **Technical Procedures:** Asks for step-by-step instructions, troubleshooting, or "how-to" guides (e.g., "How to replace a CPU?", "My code is throwing a NullPointerException").
    * **Recent Events or Data:** Refers to events after your last knowledge update or requires real-time information (e.g., "What were the latest tech stock trends this week?").
    * **Safety-Critical Information:** Answering incorrectly could have negative consequences (e.g., medical advice, financial guidance, hardware modifications).
    * **Comparisons or Recommendations:** Asks for the "best" product, comparisons between items, or specific recommendations.

2.  **skip**: Choose this *only* for questions that are definitively answerable with universal, timeless general knowledge.
    * **General Knowledge:** "What is the capital of France?", "Explain the theory of relativity."
    * **Simple Greetings & Conversations:** "Hello", "How are you?"
    * **Basic Math or Logic:** "What is 5 + 7?"
    * **Creative Writing or Brainstorming:** "Write a poem about the sea."

**Your thought process:**
First, analyze the user's question for keywords related to products, technical actions, or time-sensitive topics.
Second, evaluate the risk of providing an outdated or incorrect answer based on general knowledge alone. If there is any risk, err on the side of caution.
Finally, conclude with 'retrieve' or 'skip'.

**When in doubt, always choose 'retrieve'.**

Question: "{question}"
""").partial(question=question)

    llm = ChatOpenAI(model=config.LLM_MODEL_SUPER_ROUTER, temperature=0.0)
    chain = prompt | llm | StrOutputParser()
    
    try:
        decision = chain.invoke({})
        print(f"🧠 라우터 LLM 결정: '{decision}'")
        if "retrieve" in decision.lower():
            print("🚦 라우터: 정보 검색 필요. Team 2로 이동합니다.")
            return "team2"
        else:
            print("🚦 라우터: 정보 검색 불필요. Team 2를 건너뛰고 Team 3로 이동합니다.")
            state["status"]["team2"] = "pass"
            return "team3"
    except Exception as e:
        print(f"❌ 라우터 LLM 실행 오류: {e}. 안전하게 Team 2로 보냅니다.")
        return "team2"

def create_super_graph(team1_app, team2_app, team3_app):
    """
    지능형 라우터를 포함하여 모든 팀 서브그래프를 통합합니다.
    """
    builder = StateGraph(GlobalState)

    # 1. 각 팀 서브그래프를 노드로 추가합니다.
    builder.add_node("team1", team1_app)
    builder.add_node("team2", team2_app)
    builder.add_node("team3", team3_app)

    # 2. 엣지를 연결합니다.
    builder.set_entry_point("team1")
    
    builder.add_conditional_edges(
        "team1", # 시작 노드
        retrieval_necessity_router, # 판단 함수
        { # 판단 결과에 따른 분기
            "team2": "team2",
            "team3": "team3",
            END: END
        }
    )
    
    builder.add_edge("team2", "team3")
    builder.add_edge("team3", END)

    # 3. 최종 그래프를 컴파일합니다.
    return builder.compile()
