# app.py

import streamlit as st
import uuid
from typing import Dict, Any

# --- 프로젝트 파일 임포트 ---
from state import GlobalState
from graphs.team1_graph import create_team1_graph
from graphs.team2_graph import create_team2_graph
from graphs.team3_graph import create_team3_graph
from graphs.super_graph import create_super_graph

# --- 페이지 설정 ---
st.set_page_config(
    page_title="다중 에이전트 RAG 시스템",
    page_icon="🤖",
    layout="wide"
)

# --- UI ---
st.title("🤖 LangGraph 다중 에이전트 RAG 시스템")
st.markdown("""
이 앱은 LangGraph로 구축된 다중 에이전트 질의응답 시스템입니다. 질문을 입력하면 각 팀이 단계별로 작업을 수행하여 최종 답변을 생성합니다.
- **Team 1**: 사용자 질문을 분석하고 검색 쿼리를 생성합니다.
- **Team 2**: RAG 및 웹 검색으로 정보를 수집하고 평가합니다.
- **Team 3**: 수집된 정보를 바탕으로 최종 답변을 생성하고 검수합니다.
""")

# --- LangGraph 앱 초기화 (st.cache_resource 사용) ---
@st.cache_resource
def get_graph_app():
    """
    각 팀의 서브그래프와 슈퍼그래프를 빌드하고 컴파일하여
    실행 가능한 LangGraph 애플리케이션 객체를 반환합니다.
    """
    print("🚀 다중 에이전트 RAG 시스템을 초기화합니다...")
    with st.spinner("에이전트 시스템을 준비하는 중입니다... 잠시만 기다려주세요."):
        team1_app = create_team1_graph()
        team2_app = create_team2_graph()
        team3_app = create_team3_graph()
        super_graph_app = create_super_graph(team1_app, team2_app, team3_app)
    print("✅ 시스템 준비 완료!")
    return super_graph_app

app = get_graph_app()

# --- 채팅 기록 관리를 위한 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 이전 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 메인 로직: 사용자 입력 처리 및 그래프 실행 ---
if prompt := st.chat_input("LangGraph의 주요 특징을 표로 정리해줘."):
    # 사용자 메시지를 채팅 기록에 추가하고 화면에 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 어시스턴트 응답 처리 시작
    with st.chat_message("assistant"):
        final_answer = ""
        error_message = ""
        
        # 실시간 진행 상황을 표시할 UI 요소
        progress_placeholder = st.empty()
        answer_placeholder = st.empty()

        try:
            # 그래프 실행을 위한 초기 상태 설정
            initial_state: GlobalState = {
                "user_input": prompt,
                "status": {},
                "team1_retries": 0,
                "team2_retries": 0,
                "team3_retries": 0,
            }
            thread = {"configurable": {"thread_id": st.session_state.thread_id}}

            # LangGraph 스트림을 통해 실시간 이벤트 처리
            final_state = None
            for event in app.stream(initial_state, thread, stream_mode="values"):
                final_state = event
                
                # --- 실시간 진행 상황 업데이트 ---
                progress_text = "### 🏃‍♂️ 작업 진행 상황\n"
                status = event.get("status", {})
                
                # Team 1 상태
                if "team1" in status:
                    if status["team1"] == "pass":
                        progress_text += "✅ **Team 1 (질문 분석)**: 완료\n"
                        if event.get('rag_query'):
                             progress_text += f"   - 최적 검색 쿼리: `{event['rag_query']}`\n"
                    else:
                        progress_text += "❌ **Team 1 (질문 분석)**: 실패\n"
                else:
                    progress_text += "⏳ **Team 1 (질문 분석)**: 진행 중...\n"

                # 라우터 및 Team 2 상태
                if status.get("team1") == "pass":
                    if "team2" in status:
                         if status["team2"] == "pass":
                            progress_text += "✅ **Team 2 (정보 수집)**: 완료\n"
                         else:
                            progress_text += "❌ **Team 2 (정보 수집)**: 실패\n"
                    # Team 2가 시작되기 전이거나, 건너뛴 경우
                    elif "generated_answer" not in event:
                         progress_text += "⏳ **Team 2 (정보 수집)**: 대기 중...\n"

                # Team 3 상태
                if status.get("team2") == "pass": # Team 2가 성공해야 Team 3 시작
                    if "team3" in status:
                        if status["team3"] == "pass":
                            progress_text += "✅ **Team 3 (답변 생성)**: 완료\n"
                        else:
                            progress_text += "❌ **Team 3 (답변 생성)**: 실패\n"
                    elif "generated_answer" not in event:
                        progress_text += "⏳ **Team 3 (답변 생성)**: 대기 중...\n"

                progress_placeholder.markdown(progress_text)
                
            # 최종 결과 처리
            if final_state:
                final_answer = final_state.get("generated_answer", "")
                error_message = final_state.get("error_message", "")

        except Exception as e:
            st.error(f"시스템 실행 중 예외가 발생했습니다: {e}")
            error_message = f"시스템 오류: {e}"

        # 진행 상황 UI 제거 및 최종 결과 표시
        progress_placeholder.empty()
        
        if final_answer:
            answer_placeholder.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
        elif error_message:
            st.error(f"답변 생성에 실패했습니다: {error_message}")
            st.session_state.messages.append({"role": "assistant", "content": f"실패: {error_message}"})
        else:
            st.error("알 수 없는 오류로 답변을 생성하지 못했습니다.")
            st.session_state.messages.append({"role": "assistant", "content": "알 수 없는 오류 발생"})