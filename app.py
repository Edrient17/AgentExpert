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
        with st.spinner("LangGraph가 답변을 생성 중입니다..."):

            # === 상태 초기화 ===
            initial_state = {
                "user_input": user_input.strip(),
                "q_validity": True,
                "status": {
                    "team1": "",
                    "team2": "",
                    "team3": ""
                },
                "next_node": "team1_supervisor"
            }

            # === LangGraph 실행 ===
            result = graph.invoke(initial_state)

        st.success("✅ 응답이 생성되었습니다!")

        # === 결과 출력 ===
        st.subheader("🧾 최종 답변")
        st.markdown(result.get("generated_answer", "없음"))

        st.divider()
        st.subheader("🔍 내부 상태 (디버깅용)")
        with st.expander("전체 상태 보기"):
            st.json(result)
