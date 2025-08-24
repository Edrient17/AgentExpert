import uuid
from langchain_core.messages import HumanMessage, AIMessage
import os

# --- 1. MCP 라이브러리 임포트 ---
from mcp.server.fastmcp import FastMCP

# --- 2. 기존 프로젝트 파일 임포트 ---
from graph_factory import get_graph_app
from state import AgentState

# --- 3. MCP 서버 생성 ---
# FastMCP를 사용하여 Claude와 통신할 수 있는 서버를 만듭니다.
mcp = FastMCP(
    name="My Professional RAG Agent",
    instructions="A multi-agent system made with LangGraph. It takes questions and generates professional answers.",
)

# LangGraph 앱을 미리 로드합니다.
langgraph_app = get_graph_app()

# --- 4. 기존 RAG 로직을 MCP '도구(Tool)'로 정의 ---
# @mcp.tool() 데코레이터를 사용하여 Claude가 호출할 수 있는 함수로 만듭니다.
@mcp.tool()
def ask_agent(question: str) -> str:
    """
    사용자의 질문을 받아 다중 에이전트 RAG 시스템을 실행하고 최종 답변을 문자열로 반환합니다.
    """
    print(f"--- MCP Tool 'ask_agent' 실행 ---")
    print(f"입력 질문: {question}")
    try:
        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "team1_retries": 0, "team2_retries": 0, "team3_retries": 0,
            "global_loop_count": 0, "is_simple_query": "No"
        }
        thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # LangGraph 실행
        final_state = langgraph_app.invoke(initial_state, thread)

        # 결과에서 답변 추출
        final_answer = "답변을 생성하지 못했습니다."
        if messages := final_state.get("messages"):
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    final_answer = msg.content
                    break
        
        print(f"생성된 답변: {final_answer[:100]}...")
        return final_answer

    except Exception as e:
        print(f"MCP Tool 실행 중 오류 발생: {e}")
        return f"오류가 발생하여 답변을 생성할 수 없습니다: {e}"

# --- 5. 서버 실행 ---
if __name__ == "__main__":
    # Cloud Run은 $PORT를 자동으로 제공합니다.
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = int(os.environ.get("PORT", "8080"))
    mcp.run(transport="streamable-http")