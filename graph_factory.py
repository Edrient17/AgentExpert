import functools

# --- 프로젝트 파일 임포트 ---
from graphs.team1_graph import create_team1_graph
from graphs.team2_graph import create_team2_graph
from graphs.team3_graph import create_team3_graph
from graphs.super_graph import create_super_graph

# @st.cache_resource 대신 표준 라이브러리의 캐싱 기능을 사용합니다.
# 이렇게 하면 Streamlit에 대한 의존성이 사라집니다.
@functools.lru_cache(maxsize=1)
def get_graph_app():
    """
    각 팀의 서브그래프와 슈퍼그래프를 빌드하고 컴파일하여
    실행 가능한 LangGraph 애플리케이션 객체를 반환합니다.
    이 함수는 최초 호출 시에만 실행되고 결과는 캐싱됩니다.
    """
    print("🚀 다중 에이전트 RAG 시스템을 초기화합니다...")
    
    team1_app = create_team1_graph()
    team2_app = create_team2_graph()
    team3_app = create_team3_graph()
    super_graph_app = create_super_graph(team1_app, team2_app, team3_app)

    print("✅ 시스템 준비 완료!")
    return super_graph_app