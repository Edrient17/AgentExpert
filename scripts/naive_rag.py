import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

import config
from src.prompts.loader import load_prompt_template
from src.agents.answer.generator import generate_answer
from src.agents.query.worker import process_question
from src.schema.outputs import AnswerEvaluationResult, DocEvaluationResult, QuestionEvaluationResult
from src.tools.common import format_docs
from src.tools.rag import vector_store_rag_search


def run_naive_rag(user_question: str):
    """
    Run a simplified, sequential RAG pipeline for local debugging.
    The production app uses the LangGraph supervisor in graph_factory.py.
    """
    print(f"[start] Starting naive RAG pipeline. Question: '{user_question}'")

    state = {
        "messages": [HumanMessage(content=user_question)],
        "rag_docs": [],
        "web_docs": [],
    }

    print("\n--- 1. Question processing (Query Worker) ---")
    t1_result = process_question(state)
    t1_message = t1_result["messages"][0]
    state["messages"].append(t1_message)

    processed_data = t1_message.additional_kwargs
    q_en_transformed = processed_data.get("q_en_transformed", "")
    rag_queries = processed_data.get("rag_queries", [])
    output_format = processed_data.get("output_format", ["qa", "ko"])

    if not rag_queries:
        print("[error] Query Worker did not generate RAG queries. Stopping pipeline.")
        return

    first_rag_query = rag_queries[0]
    state["q_en_transformed"] = q_en_transformed
    state["output_format"] = output_format
    state["best_rag_query"] = first_rag_query

    print(f"[ok] Refined question: {q_en_transformed}")
    print(f"[ok] RAG query: '{first_rag_query}'")

    print("\n--- 2. Question evaluation (Query Evaluator) ---")
    t1_parser = JsonOutputParser(p_object=QuestionEvaluationResult)
    t1_prompt = load_prompt_template("query/question_evaluator.yaml")
    t1_llm = ChatOpenAI(
        model=config.LLM_MODEL_TEAM1,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    t1_chain = t1_prompt.partial(schema=t1_parser.get_format_instructions()) | t1_llm | t1_parser
    try:
        t1_eval_dict = t1_chain.invoke(
            {
                "user_input": user_question,
                "q_en_transformed": q_en_transformed,
                "output_format": json.dumps(output_format, ensure_ascii=False),
                "default_format": json.dumps(["qa", "ko"], ensure_ascii=False),
                "rag_queries_json": json.dumps(rag_queries, ensure_ascii=False),
            }
        )
        print("[ok] Query Evaluator completed:")
        print(json.dumps(t1_eval_dict, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[error] Query Evaluator failed: {e}")

    print("\n--- 3. Document retrieval (Retrieval RAG Worker) ---")
    try:
        rag_docs = vector_store_rag_search.func(query=first_rag_query, top_k=5, rerank_k=5)
        state["rag_docs"] = rag_docs
        print(f"[ok] Retrieved {len(rag_docs)} documents from the RAG store.")
    except Exception as e:
        print(f"[error] Document retrieval failed: {e}. Stopping pipeline.")
        return

    print("\n--- 4. Retrieved document evaluation (Retrieval Evaluator) ---")
    t2_parser = JsonOutputParser(p_object=DocEvaluationResult)
    t2_prompt = load_prompt_template("retrieval/document_evaluator.yaml")
    t2_llm = ChatOpenAI(
        model=config.LLM_MODEL_TEAM2_EVAL,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    t2_chain = t2_prompt.partial(schema=t2_parser.get_format_instructions()) | t2_llm | t2_parser

    for i, doc in enumerate(rag_docs):
        try:
            preview = (getattr(doc, "page_content", "") or "")[:4000]
            t2_eval_dict = t2_chain.invoke(
                {
                    "q_en_transformed": q_en_transformed,
                    "rag_query": first_rag_query,
                    "doc_text": preview,
                }
            )
            print(f"  - Document #{i + 1} evaluation: {json.dumps(t2_eval_dict, ensure_ascii=False)}")
        except Exception as e:
            print(f"  - Document #{i + 1} evaluation failed: {e}")

    print("\n--- 5. Answer generation (Answer Generator) ---")
    t3_gen_result = generate_answer(state)
    final_answer_msg = t3_gen_result["messages"][0]
    state["messages"].append(final_answer_msg)

    print("[ok] Final answer generated.")
    print("=" * 30)
    print(final_answer_msg.content)
    print("=" * 30)

    print("\n--- 6. Final answer evaluation (Answer Evaluator) ---")
    t3_parser = JsonOutputParser(p_object=AnswerEvaluationResult)
    t3_prompt = load_prompt_template("answer/evaluator.yaml")
    t3_llm = ChatOpenAI(
        model=config.LLM_MODEL_TEAM3_EVAL,
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    t3_chain = t3_prompt.partial(schema=t3_parser.get_format_instructions()) | t3_llm | t3_parser

    try:
        t3_eval_dict = t3_chain.invoke(
            {
                "q_en_transformed": q_en_transformed,
                "output_format": json.dumps(output_format, ensure_ascii=False),
                "generated_answer": final_answer_msg.content,
                "retrieved_docs": format_docs(state["rag_docs"]),
            }
        )
        print("[ok] Answer Evaluator completed:")
        print(json.dumps(t3_eval_dict, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[error] Answer evaluation failed: {e}")


if __name__ == "__main__":
    sample_question = "Summarize the major camera features of the Galaxy S24."
    run_naive_rag(sample_question)
