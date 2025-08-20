import streamlit as st
import uuid
from typing import Dict, Any, List

# --- 프로젝트 파일 임포트 ---
from state import AgentState # 변경된 State
from graphs.team1_graph import create_team1_graph
from graphs.team2_graph import create_team2_graph
from graphs.team3_graph import create_team3_graph
from graphs.super_graph import create_super_graph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

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

# --- Helper Function for Progress Tracking ---
def parse_progress(messages: List[Dict[str, Any]]) -> str:
    """메시지 목록을 분석하여 실시간 진행 상황 텍스트를 생성합니다."""
    progress_text = "### 🏃‍♂️ 작업 진행 상황\n"
    team1_status, team2_status, team3_status = "⏳ 진행 중...", "⏳ 대기 중...", "⏳ 대기 중..."
    rag_query = ""
    team1_failed = False

    for msg in messages:
        # Team 1 상태 분석
        if msg.name == "team1_evaluator":
            if msg.content == "pass":
                team1_status = "✅ 완료"
                rag_query = msg.additional_kwargs.get("best_rag_query", "")
            else:
                team1_status = f"❌ 실패 ({msg.content})"
                team1_failed = True
    
    progress_text += f"**Team 1 (질문 분석)**: {team1_status}\n"
    if rag_query:
        progress_text += f"   - 최적 검색 쿼리: `{rag_query}`\n"

    if team1_status == "✅ 완료":
        team2_started = any(m.name in ["rag_search_result", "web_search_result"] for m in messages)
        team2_evaluated = any(m.name == "team2_evaluator" for m in messages)

        if not team2_started:
             team2_status = "⏳ 진행 중..." # 라우터가 Team3로 바로 보낼 수도 있음
        
        if team2_evaluated:
            team2_eval_msg = next((m for m in reversed(messages) if m.name == "team2_evaluator"), None)
            if team2_eval_msg and team2_eval_msg.content == "pass":
                team2_status = "✅ 완료"
            else:
                team2_status = f"❌ 실패 ({team2_eval_msg.content if team2_eval_msg else 'N/A'})"
        
        progress_text += f"**Team 2 (정보 수집)**: {team2_status}\n"

        if team2_status == "✅ 완료":
            team3_evaluated = any(m.name == "final_evaluator" for m in messages)
            if not team3_evaluated:
                team3_status = "⏳ 진행 중..."
            else:
                team3_eval_msg = next((m for m in reversed(messages) if m.name == "final_evaluator"), None)
                if team3_eval_msg and team3_eval_msg.content == "pass":
                    team3_status = "✅ 완료"
                else:
                    team3_status = f"❌ 실패 ({team3_eval_msg.content if team3_eval_msg else 'N/A'})"
            
            progress_text += f"**Team 3 (답변 생성)**: {team3_status}\n"

    elif team1_failed:
        progress_text += "**Team 2 (정보 수집)**: 🛑 중단\n"
        progress_text += "**Team 3 (답변 생성)**: 🛑 중단\n"


    return progress_text

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
            # 그래프 실행을 위한 초기 상태 설정 (메시지 기반)
            initial_state: AgentState = {
                "messages": [HumanMessage(content=prompt)],
                "team1_retries": 0,
                "team2_retries": 0,
                "team3_retries": 0,
            }
            thread = {"configurable": {"thread_id": st.session_state.thread_id}}

            # LangGraph 스트림을 통해 실시간 이벤트 처리
            final_state_messages = []
            for event in app.stream(initial_state, thread, stream_mode="values"):
                # 이벤트에서 메시지 목록을 가져옴
                messages = event.get("messages", [])
                final_state_messages = messages
                
                # 메시지 목록을 분석하여 진행 상황 업데이트
                progress_text = parse_progress(messages)
                progress_placeholder.markdown(progress_text)
                
            # 최종 결과 처리
            if final_state_messages:
                # 마지막 AIMessage를 최종 답변으로 간주
                for msg in reversed(final_state_messages):
                    if isinstance(msg, AIMessage):
                        final_answer = msg.content
                        break
                
                # 최종 답변을 찾지 못한 경우, 마지막 메시지에서 에러 확인
                if not final_answer:
                    last_msg = final_state_messages[-1]
                    if last_msg.content != "pass":
                        error_message = f"워크플로우가 실패로 종료되었습니다. (마지막 단계: {last_msg.name}, 이유: {last_msg.content})"

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
