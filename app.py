import streamlit as st
import uuid
from typing import Dict, Any, List
import re
import os 

# --- 프로젝트 파일 임포트 ---
from state import AgentState
from graph_factory import get_graph_app
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# --- 페이지 설정 ---
st.set_page_config(
    page_title="Agent Expert",
    page_icon="🤖",
    layout="wide"
)

# --- UI ---
st.title("🤖 Agent Expert")
st.markdown("""
이 앱은 LangGraph로 구축된 멀티 에이전트 Q&A 시스템입니다. 질문을 입력하면 각 팀이 단계별로 작업을 수행하여 최종 답변을 생성합니다.
- **Team Query**: 사용자 질문을 분석하고 검색 쿼리를 생성합니다.
- **Team Search**: RAG 및 웹 검색으로 정보를 수집하고 평가합니다.
- **Team Answer**: 수집된 정보를 바탕으로 최종 답변을 생성하고 검수합니다.
""")

# --- LangGraph 앱 로드 ---
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
    team1_status, team2_status, team3_status = "⏳ 질문 분석 중...\n", "⏳ 대기 중...\n", "⏳ 대기 중...\n"
    rag_query = ""
    team1_failed = False

    for msg in messages:
        # Team Query 상태 분석
        if msg.name == "team1_evaluator":
            if msg.content == "pass":
                team1_status = "✅ 완료\n"
                rag_query = msg.additional_kwargs.get("best_rag_query", "")
            else:
                team1_status = f"❌ 실패 ({msg.content})\n"
                team1_failed = True
    
    progress_text += f"**Team Query (질문 분석)**: {team1_status}\n"
    if rag_query:
        progress_text += f"   - 최적 검색 쿼리: `{rag_query}`\n\n"

    if team1_status == "✅ 완료\n":
        team2_started = any(m.name in ["rag_search_result", "web_search_result"] for m in messages)
        team2_evaluated = any(m.name == "team2_evaluator" for m in messages)

        if not team2_started:
             team2_status = "⏳ 데이터 수집 진행 중...\n" # 라우터가 Team3로 바로 보낼 수도 있음
        
        if team2_evaluated:
            team2_eval_msg = next((m for m in reversed(messages) if m.name == "team2_evaluator"), None)
            if team2_eval_msg and team2_eval_msg.content == "pass":
                team2_status = "✅ 완료\n"
            else:
                team2_status = f"❌ 실패 ({team2_eval_msg.content if team2_eval_msg else 'N/A'})\n"
        
        progress_text += f"**Team Search (정보 수집)**: {team2_status}\n"

        if team2_status == "✅ 완료\n":
            team3_evaluated = any(m.name == "final_evaluator" for m in messages)
            if not team3_evaluated:
                team3_status = "⏳ 답변 생성 중...\n"
            else:
                team3_eval_msg = next((m for m in reversed(messages) if m.name == "final_evaluator"), None)
                if team3_eval_msg and team3_eval_msg.content == "pass":
                    team3_status = "✅ 완료\n"
                else:
                    team3_status = f"❌ 실패 ({team3_eval_msg.content if team3_eval_msg else 'N/A'})\n"
            
            progress_text += f"**Team Answer (답변 생성)**: {team3_status}\n"

    elif team1_failed:
        progress_text += "**Team Search (정보 수집)**: 🛑 중단\n\n"
        progress_text += "**Team Answer (답변 생성)**: 🛑 중단\n\n"


    return progress_text

# --- 메인 로직: 사용자 입력 처리 및 그래프 실행 ---
if prompt := st.chat_input("질문을 입력해주세요."):
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
                "global_loop_count": 0,
                "is_simple_query": "No"
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
            # 답변에서 마크다운 테이블과 이미지 경로를 분리
            image_path_marker = "**[생성된 표 이미지 보기]"
            
            if image_path_marker in final_answer:
                parts = final_answer.split(image_path_marker)
                markdown_content = parts[0].strip()
                link_part = parts[1]

                # 정규표현식으로 괄호 안의 파일 경로 추출
                match = re.search(r'\((.*?)\)', link_part)
                if match:
                    image_path = match.group(1)
                    
                    # 1. 마크다운 테이블 표시
                    answer_placeholder.markdown(markdown_content)
                    
                    # 2. 추출된 경로의 이미지를 st.image로 표시
                    if os.path.exists(image_path):
                        st.image(image_path, caption="생성된 표 이미지")
                    else:
                        st.warning(f"이미지 파일을 찾을 수 없습니다: {image_path}")
                else:
                    # 링크 형식이 잘못된 경우, 원본 전체를 표시
                    answer_placeholder.markdown(final_answer)
            else:
                # 테이블 이미지가 없는 일반 답변은 그대로 표시
                answer_placeholder.markdown(final_answer)

            st.session_state.messages.append({"role": "assistant", "content": final_answer})
        elif error_message:
            st.error(f"답변 생성에 실패했습니다: {error_message}")
            st.session_state.messages.append({"role": "assistant", "content": f"실패: {error_message}"})
        else:
            st.error("알 수 없는 오류로 답변을 생성하지 못했습니다.")
            st.session_state.messages.append({"role": "assistant", "content": "알 수 없는 오류 발생"})
