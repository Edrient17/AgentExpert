# app.py
import os
import streamlit as st
from langchain_core.tracers import LangChainTracer
from dotenv import load_dotenv
from main_graph import graph

load_dotenv()
tracer = LangChainTracer(project_name=os.getenv("LANGCHAIN_PROJECT", "AgentExpert"))

# === 초기화 ===
st.set_page_config(page_title="LangGraph RAG QA", layout="wide")
st.title("LangGraph 기반 질문 응답 시스템")

# === 입력 받기 ===
user_input = st.text_area("💬 질문을 입력하세요", height=100)

if st.button("질문 실행"):
    if not user_input.strip():
        st.warning("질문을 입력해주세요.")
    else:
        with st.spinner("답변을 생성 중입니다..."):

            # === 상태 초기화 ===
            initial_state = {
                # 입력
                "user_input": user_input.strip(),

                # 기본값
                "q_validity": True,
                "q_en_transformed": "",
                "rag_queries": [],
                "rag_query": "",
                "rag_query_scores": [],
                "output_format": ["qa", "ko"],

                # 후속 단계
                "rag_docs": [],
                "web_docs": [],
                "generated_answer": "",

                # 에러/제어
                "error_message": "",
                "status": {"team1": "", "team2": "", "team3": ""},
                "next_node": "team1_supervisor",
            }

            # === LangGraph 실행 ===
            result = graph.invoke(initial_state)

        # === 에러 표시 or 최종 답변 표시 ===
        if result.get("error_message"):
            st.error(f"오류가 발생했습니다.: {result['error_message']}")
        else:
            st.subheader("🧾 최종 답변")
            answer = result.get("generated_answer", "").strip()
            st.markdown(answer or "_생성된 답변이 없습니다._")

        # === 내부 상태 (디버깅용) ===
        st.divider()
        with st.expander("🧩 전체 상태 보기 (디버깅)"):
            st.json(result)
