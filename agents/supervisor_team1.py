import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from agents.team1_query_agent import agent_team1_question_processing
from langchain_core.runnables import RunnableLambda

EVAL_PROMPT = """
다음은 사용자 질문과 Agent가 분석한 결과입니다.

[User Input]
{user_input}

[q_validity]
{q_validity}

[q_en_transformed]
{q_en_transformed}

[rag_query]
{rag_query}

이 세 결과가 질문 의미를 잘 반영하고 논리적으로 정합한지 평가해주세요.
1. semantic_alignment 점수 (0~1)로 평가

아래 형식으로 출력하세요:
{{ "semantic_alignment": float }}
"""

def supervisor_team1(state: dict) -> dict:
    print(f"🔍 Team 1 Supervisor 시작")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate.from_template(EVAL_PROMPT)
    chain = prompt | llm

    max_attempts = 2
    current_attempts = 0  # state에서 attempts 제거

    while current_attempts < max_attempts:
        print(f"⚙️ Team 1 Agent 실행 (시도 {current_attempts + 1})")
        state = RunnableLambda(agent_team1_question_processing).invoke(state)

        if not state.get("q_validity", False):
            print("❌ 질문 유효성 실패")
            current_attempts += 1
            if current_attempts >= max_attempts:
                print("❌ 최대 시도 도달 → 실패")
                state["status"]["team1"] = "fail"
                break
            else:
                print("🔁 유효성 실패 → 재시도")
                continue

        # LLM 평가 시도
        try:
            result = chain.invoke({
                "user_input": state["user_input"],
                "q_validity": str(state["q_validity"]),
                "q_en_transformed": state["q_en_transformed"],
                "rag_query": state["rag_query"]
            })

            parsed = json.loads(result.content)
            score = parsed.get("semantic_alignment", 0.0)

            if score >= 0.5:
                print(f"✅ Team 1 통과 (score={score:.2f})")
                state["status"]["team1"] = "pass"
                break
            else:
                print(f"🔁 평가 실패 (score={score:.2f}) → 재시도")
                current_attempts += 1

        except Exception as e:
            print(f"❌ 평가 중 오류 발생: {e}")
            current_attempts += 1

    if current_attempts >= max_attempts and state["status"].get("team1") != "pass":
        print("❌ 최종 실패 → Team 1 fail 처리")
        state["status"]["team1"] = "fail"

    return state
