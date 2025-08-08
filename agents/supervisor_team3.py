import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from agents.team3_answer_agent import agent_team3_answer_generation  # 반드시 import

PROMPT = """
[질문]
{question}

[답변]
{answer}

이 답변이 질문에 의미적으로 적절하고, 신뢰할 수 있는지를 종합적으로 판단해주세요.

- answer_quality_score (0~1): 질문과의 정합성, 신뢰도, 명확성을 종합한 점수

아래 JSON 형식으로 출력하세요:

{{
  "answer_quality_score": float
}}
"""

def supervisor_team3(state: dict) -> dict:
    print("🔍 Team 3 Supervisor 시작")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate.from_template(PROMPT)
    chain = prompt | llm

    max_attempts = 2
    current_attempts = 0  # attempts state 참조 없이 내부에서만 카운트

    while current_attempts < max_attempts:
        print(f"⚙️ Team 3 Agent 실행 (시도 {current_attempts + 1})")
        state = RunnableLambda(agent_team3_answer_generation).invoke(state)

        question = state.get("q_en_transformed", "").strip()
        answer = state.get("generated_answer", "").strip()

        if not question or not answer:
            print("❌ 질문 또는 답변 누락")
            current_attempts += 1
            if current_attempts >= max_attempts:
                print("❌ 최대 시도 도달 → 실패")
                state["status"]["team3"] = "fail"
                break
            else:
                print("🔁 누락 → 재시도")
                continue

        try:
            result = chain.invoke({
                "question": question,
                "answer": answer
            })

            parsed = json.loads(result.content)
            score = parsed.get("answer_quality_score", 0.0)

            if score >= 0.5:
                print(f"✅ Team 3 통과 (score={score:.2f})")
                state["status"]["team3"] = "pass"
                break
            else:
                print(f"🔁 평가 실패 (score={score:.2f}) → 재시도")
                current_attempts += 1

        except Exception as e:
            print(f"❌ 평가 중 오류 발생: {e}")
            current_attempts += 1

    if current_attempts >= max_attempts and state["status"].get("team3") != "pass":
        print("❌ 최종 실패 → Team 3 fail 처리")
        state["status"]["team3"] = "fail"

    return state
