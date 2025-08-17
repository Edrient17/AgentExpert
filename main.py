# main.py

import uuid
from state import GlobalState
from graphs.team1_graph import create_team1_graph
from graphs.team2_graph import create_team2_graph
from graphs.team3_graph import create_team3_graph
from graphs.super_graph import create_super_graph


if __name__ == "__main__":
    print("🚀 다중 에이전트 RAG 시스템 리팩토링을 시작합니다.")

    # 1. 각 팀의 서브그래프(app)를 생성합니다.
    print("🌀 1. 각 팀의 서브그래프를 생성합니다...")
    team1_app = create_team1_graph()
    team2_app = create_team2_graph()
    team3_app = create_team3_graph()
    print("✅ 서브그래프 생성 완료!")

    # 2. 생성된 서브그래프들을 슈퍼그래프에 통합합니다.
    print("🌐 2. 서브그래프들을 슈퍼그래프로 통합합니다...")
    super_graph_app = create_super_graph(team1_app, team2_app, team3_app)
    print("✅ 슈퍼그래프 통합 완료!")

    # 3. 사용자 질문으로 시스템을 실행합니다.
    print("\n💬 3. 사용자 질문으로 시스템 실행을 시작합니다...")
    user_input = "LangGraph에 대해 설명해주고, 주요 특징을 표로 정리해줘."
    initial_state: GlobalState = {"user_input": user_input, "status": {}}
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

    final_state = None
    for event in super_graph_app.stream(initial_state, thread):
        final_state = event

    # 4. 최종 결과를 출력합니다.
    print("\n✨ 4. 최종 결과:")
    if final_state:
        print(final_state.get("generated_answer", "답변을 생성하지 못했습니다."))
    else:
        print("오류: 최종 상태에 도달하지 못했습니다.")