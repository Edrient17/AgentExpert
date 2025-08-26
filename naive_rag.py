import json
from langchain_core.messages import HumanMessage

from agents.team1_agents import process_question, QuestionEvaluationResult
from agents.team2_agents import DocEvaluationResult
from agents.team3_agents import generate_answer
from utility_tools import vector_store_rag_search, format_docs
import config

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class AnswerEvaluationResult(BaseModel):
    rules_compliance: bool
    question_coverage: float
    hallucination_score: float
    error_message: str = ""

def run_naive_rag(user_question: str):
    """
    모든 단계의 평가를 포함하는 단순화된 순차적 RAG 파이프라인을 실행합니다.
    """
    print(f"🚀 Naive RAG 파이프라인 (전체 평가 포함)을 시작합니다. 질문: '{user_question}'")

    # --- 1. 초기 상태 설정 ---
    state = {
        "messages": [HumanMessage(content=user_question)],
        "rag_docs": [],
        "web_docs": []
    }

    # --- 2. Team1 Worker: 질문 처리 ---
    print("\n--- 1. 질문 처리 (Team 1) ---")
    t1_result = process_question(state)
    t1_message = t1_result['messages'][0]
    state['messages'].append(t1_message)

    processed_data = t1_message.additional_kwargs
    q_en_transformed = processed_data.get("q_en_transformed", "")
    rag_queries = processed_data.get("rag_queries", [])
    output_format = processed_data.get("output_format", ["qa", "ko"])

    if not rag_queries:
        print("❌ Team 1에서 RAG 쿼리를 생성하지 못했습니다. 프로세스를 종료합니다.")
        return

    first_rag_query = rag_queries[0]
    state['q_en_transformed'] = q_en_transformed
    state['output_format'] = output_format
    state['best_rag_query'] = first_rag_query

    print(f"✅ 처리된 질문: {q_en_transformed}")
    print(f"✅ 사용할 RAG 쿼리: '{first_rag_query}'")

    # --- 3. Team1 Evaluator: 질문 처리 결과 평가 (점수 측정 전용) ---
    print("\n--- 2. 질문 처리 평가 (Team 1) ---")
    t1_parser = JsonOutputParser(p_object=QuestionEvaluationResult)
    t1_prompt = PromptTemplate.from_template("""
You are the Team1 Supervisor evaluator. Using the information below, make binary judgments and per-query scores.

[User Input]
{user_input}

[q_en_transformed]  # refined English question from the agent
{q_en_transformed}

[output_format]  # [type, language]
{output_format}

[default_format]
{default_format}

[rag_queries]
{rag_queries_json}

[Scoring Guide]

You must evaluate the document using the following criteria.
You must select only one of the predefined values. Do not output any intermediate values.

1) semantic_alignment (float in [0,1])  
   - Definition: How accurately `q_en_transformed` reflects the meaning and constraints of `user_input`.  
   - Allowed values: {{0.00, 0.25, 0.50, 0.75, 1.00}}  
     * 0.00 (None): completely irrelevant or incorrect  
     * 0.25 (Low): only some keywords overlap, misses the core meaning/constraints  
     * 0.50 (Partial): captures main meaning but lacks important constraints or nuance  
     * 0.75 (Strong): satisfies most meaning/constraints with solid coverage, but small gaps remain  
     * 1.00 (Exact): fully faithful to meaning/constraints, immediately usable  

2) format_compliance (bool)  
   - Definition: Whether the `output_format` matches the user’s requested format.  
   - Decision process:  
     a) Analyze `user_input`. Did the user explicitly request a format/language (e.g., "표로", "영어로", "in a table", "in English")?  
     b) If YES → `format_compliance = true` only if `output_format` exactly matches the user’s request (ignore default_format).  
     c) If NO → `format_compliance = true` only if `output_format` == `default_format`.  

3) rag_query_scores (list[float])  
   - Definition: For each `rag_query`, score how well it captures the user’s requirements (keywords, constraints, time ranges, numbers/units, search-friendliness).  
   - Each value must be in [0.00, 1.00], continuous scale.  
   - The length MUST equal len(rag_queries).  

4) error_message (str)  
   - If the document is empty, irrelevant, too generic, or duplicated → return a short note in Korean.  
   - Otherwise return "" (empty string).
                                          
Return JSON ONLY. Do not include any additional text.

Output schema:
{schema}
""").partial(schema=t1_parser.get_format_instructions())
    t1_llm = ChatOpenAI(model=config.LLM_MODEL_TEAM1, temperature=0.0, model_kwargs={"response_format": {"type": "json_object"}})
    t1_chain = t1_prompt | t1_llm | t1_parser
    try:
        t1_eval_dict = t1_chain.invoke({
            "user_input": user_question,
            "q_en_transformed": q_en_transformed,
            "output_format": json.dumps(output_format, ensure_ascii=False),
            "default_format": json.dumps(["qa", "ko"], ensure_ascii=False),
            "rag_queries_json": json.dumps(rag_queries, ensure_ascii=False)
        })
        print("✅ Team 1 평가 완료:")
        print(json.dumps(t1_eval_dict, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌ Team 1 평가 중 오류 발생: {e}")

    # --- 4. Team2 Worker: 문서 검색 (도구를 직접 호출) ---
    print("\n--- 3. 문서 검색 (Team 2) ---")
    try:
        rag_docs = vector_store_rag_search.func(query=first_rag_query, top_k=5, rerank_k=5)
        state['rag_docs'] = rag_docs
        print(f"✅ {len(rag_docs)}개의 문서를 RAG 스토어에서 검색했습니다.")
    except Exception as e:
        print(f"❌ 문서 검색 중 오류 발생: {e}. 프로세스를 종료합니다.")
        return

    # --- 5. Team2 Evaluator: 검색 문서 평가 (점수 측정 전용) ---
    print("\n--- 4. 검색 문서 평가 (Team 2) ---")
    t2_parser = JsonOutputParser(p_object=DocEvaluationResult)
    t2_prompt = PromptTemplate.from_template("""
You are a strict Quality Control Supervisor evaluator.
Your job is to carefully assess whether the given document is sufficiently relevant and detailed to help answer the user’s question.
Follow the instructions below without deviation and return ONLY a valid JSON object matching the schema.

[Evaluation Guidelines]
- Use the two scoring rubrics below independently: one for semantic_relevance and one for is_detailed.
- Judge only based on the provided inputs. Do not invent information.
- All contents of the document should be considered, not just part of the document.

[Scoring Guide — semantic_relevance]
Choose EXACTLY one level for how well the document matches the question’s intent and constraints (subject, entities, context).
- 0.00 = None: completely irrelevant or empty; contradicts the question or ignores core entities/constraints.
- 0.25 = Low: superficial keyword overlap; misses main intent or key constraints; noticeable topic drift.
- 0.50 = Partial: addresses the main topic but misses important constraints/context; mixed or uneven relevance.
- 0.75 = Strong: satisfies most intent/constraints with minor mismatches or small gaps.
- 1.00 = Exact: fully aligned with the question’s entities and constraints; no topic drift.

[Scoring Guide — is_detailed]
Choose EXACTLY one level for how specific and sufficient the document is to support a reliable answer.
- 0.00 = None: empty/generic; no actionable specifics.
- 0.25 = Low: few specifics; vague statements; lacks steps, data, or concrete facts.
- 0.50 = Partial: some specifics but missing key details to answer fully; incomplete coverage.
- 0.75 = Strong: solid specifics; covers most needed details with minor gaps.
- 1.00 = Exact: comprehensive and specific (e.g., steps, parameters, examples, citations, or numbers); fully sufficient.

[Error Message]
- If the document is empty, irrelevant, duplicated, off-topic, or too generic/outdated for the question, write a short English note (<= 20 words).
- Otherwise, return "".

[Inputs]
- Question Summary: {q_en_transformed}
- RAG Query: {rag_query}
- Document (excerpted for evaluation): {doc_text}

[Output Instructions]
- Return ONLY a valid JSON object; no commentary, Markdown, code fences, or extra keys.
- Keys must exactly match the schema fields.
- Values for the two scores MUST be one of: 0.00, 0.25, 0.50, 0.75, 1.00.

Output schema:
{schema}
""").partial(schema=t2_parser.get_format_instructions())
    t2_llm = ChatOpenAI(model=config.LLM_MODEL_TEAM2_EVAL, temperature=0.0, model_kwargs={"response_format": {"type": "json_object"}})
    t2_chain = t2_prompt | t2_llm | t2_parser

    for i, doc in enumerate(rag_docs):
        try:
            preview = (getattr(doc, "page_content", "") or "")[:4000]
            t2_eval_dict = t2_chain.invoke({"q_en_transformed": q_en_transformed, "rag_query": first_rag_query, "doc_text": preview})
            print(f"  - 문서 #{i+1} 평가 점수: {json.dumps(t2_eval_dict, ensure_ascii=False)}")
        except Exception as e:
            print(f"  - 문서 #{i+1} 평가 중 오류 발생: {e}")
    print("✅ 모든 검색 문서에 대한 개별 평가를 완료했습니다.")

    # --- 6. Team3 Worker: 답변 생성 ---
    print("\n--- 5. 답변 생성 (Team 3) ---")
    t3_gen_result = generate_answer(state)
    final_answer_msg = t3_gen_result['messages'][0]
    state['messages'].append(final_answer_msg)

    print("✅ 최종 답변이 생성되었습니다.")
    print("="*30)
    print(final_answer_msg.content)
    print("="*30)

    # --- 7. Team3 Evaluator: 최종 답변 평가 (점수 측정 전용) ---
    print("\n--- 6. 최종 답변 평가 (Team 3) ---")
    t3_parser = JsonOutputParser(p_object=AnswerEvaluationResult)
    t3_prompt = PromptTemplate.from_template("""
You are the Team 3 Supervisor evaluator. Judge the final answer against the requested format,
the refined question, AND the provided documents.

Inputs:
[Refined question]
{q_en_transformed}

[Output format]  # ["type", "language"]
{output_format}

[Generated answer]
{generated_answer}

[Retrieved docs]
{retrieved_docs}

Scoring policy (deterministic):
- For EACH criterion, choose ONE value from {{0, 0.25, 0.50, 0.75, 1.00}}.
- Use the anchor descriptions below. If borderline, ROUND DOWN to the nearest anchor.
- Provide a concise error_message highlighting the single most limiting issue when ANY score < 0.75; otherwise use an empty string.
- Return JSON ONLY with the EXACT keys (no extra or missing): rules_compliance, question_coverage, hallucination_score, error_message.
- If any score would be N/A, still include the key with 0.00.
- If the [Generated answer] contains a Markdown table AND a link to an image file (e.g., `[생성된 표 이미지 보기](...)`), you MUST IGNORE the image link part for your evaluation. Your assessment should be based solely on the content and structure of the Markdown table itself.

Criteria & anchors:

1) rules_compliance (float): Adherence to hard rules (requested type/language/structure; no guessing beyond docs; no URLs/academic-style citations; no think/reasoning reveal; correct subtle attribution if Web info is used).
   - 1.00: Fully matches requested type & language; respects all hard rules; attribution phrased correctly when needed and if Web Passages were used, the answer clearly signals web origin once (inline or closing note), phrased naturally without URLs/numeric citations.
   - 0.75: Minor format/tone slips but type/language correct; no safety/grounding violations.
   - 0.50: Multiple minor issues or one significant format error (e.g., partially wrong structure) but salvageable and Web Passages used but no explicit web attribution; or multiple redundant attributions.
   - 0.25: Major type/structure mismatch or language mix; partial rule breaches (e.g., informal citations) without safety breach.
   - 0.00: Ignores requested type/language or violates hard rules (e.g., reveals reasoning, adds URLs/academic citations, unsafe content) and Uses URLs/numeric citations/footnotes for web attribution; or mislabels the source.

2) question_coverage (float): Degree to which the answer addresses the refined question (intent, scope, constraints, sub-questions).
   - 1.00: Covers all core requirements, constraints, and sub-parts; anticipates edge cases as appropriate.
   - 0.75: Covers the main intent and most constraints; minor omissions that don’t affect utility.
   - 0.50: Partial coverage; misses at least one key requirement or constraint.
   - 0.25: Largely off-target; touches the topic but not the user’s actual need.
   - 0.00: Irrelevant or fails to address the question.

3) hallucination_score (float): Groundedness in retrieved docs (RAG/Web); correct application of “RAG takes precedence”; proper subtle attribution when Web info is used.
   - 1.00: All claims directly supported; no contradictions; attribution used correctly when needed.
   - 0.75: Vast majority grounded; minor harmless inferences; no contradictions.
   - 0.50: Several claims weakly/implicitly supported or missing clear grounding.
   - 0.25: Many claims unsupported; suspected fabrication.
   - 0.00: Mostly conjecture or contradicts the documents.

Return JSON ONLY with:
{schema}
""").partial(schema=t3_parser.get_format_instructions())
    t3_llm = ChatOpenAI(model=config.LLM_MODEL_TEAM3_EVAL, temperature=0.0, model_kwargs={"response_format": {"type": "json_object"}})
    t3_chain = t3_prompt | t3_llm | t3_parser

    try:
        t3_eval_dict = t3_chain.invoke({
            "q_en_transformed": q_en_transformed,
            "output_format": json.dumps(output_format, ensure_ascii=False),
            "generated_answer": final_answer_msg.content,
            "retrieved_docs": format_docs(state["rag_docs"])
        })
        print("✅ 최종 답변 평가 완료:")
        print(json.dumps(t3_eval_dict, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"❌ 답변 평가 중 오류 발생: {e}")

#####################################################################################
if __name__ == "__main__":
    sample_question = "갤럭시 S24의 주요 카메라 기능들을 정리해줘." # 질문을 여기에 입력해주세요!!!!!!
    run_naive_rag(sample_question)