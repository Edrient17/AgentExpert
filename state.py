# state.py

from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """
    전체 워크플로우의 상태를 메시지 목록으로 관리합니다.
    """
    messages: Annotated[List[BaseMessage], operator.add]

    # --- 매니저-워커 제어 흐름 ---
    # 매니저가 특정 팀에게 수정을 요청할 때 사용하는 피드백
    manager_feedback: Optional[str]
    # 매니저가 다음에 호출할 팀을 지정
    next_team_to_call: str

    # 제어 흐름을 위한 최소한의 상태 (재시도 횟수)
    team1_retries: int
    team2_retries: int
    team3_retries: int

    global_loop_count: int