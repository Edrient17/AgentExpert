def chief_supervisor_router(state: dict) -> dict:
    """
    Chief Supervisor:
    - 각 팀의 평가 상태를 보고 다음 supervisor 지시를 next_node에 저장
    """
    status = state.get("status", {})

    # 팀1 처리
    if status.get("team1") == "fail":
        state["next_node"] = "end"
    elif status.get("team1") != "pass":
        state["next_node"] = "team1_supervisor"

    # 팀2 처리
    elif status.get("team2") == "fail":
        state["next_node"] = "end"
    elif status.get("team2") != "pass":
        state["next_node"] = "team2_supervisor"

    # 팀3 처리
    elif status.get("team3") == "fail":
        state["next_node"] = "end"
    elif status.get("team3") != "pass":
        state["next_node"] = "team3_supervisor"

    else:
        print("프로그램 종료")
        state["next_node"] = "end"

    return state