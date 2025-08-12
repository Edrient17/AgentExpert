# agents/team3_answer_agent.py
from typing import List, Union, Dict, Any
import os

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# ===== Lazy LLM =====
def _get_llm() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ===== Docs util (same pattern as Team2) =====
def _docs_preview(docs: List[Union[str, object]], max_chars: int = 10000) -> str:
    """LangChain Document|str 목록을 텍스트로 결합하고 길이를 제한."""
    parts: List[str] = []
    for d in docs:
        try:
            text = getattr(d, "page_content", None)
            if text is None and isinstance(d, str):
                text = d
            if text:
                t = str(text).strip()
                if t:
                    parts.append(t)
        except Exception:
            continue
    joined = "\n\n---\n\n".join(parts)
    if not joined:
        return "[NO CONTENT]"
    if len(joined) > max_chars:
        return joined[:max_chars] + "\n\n...[truncated]..."
    return joined

# ===== Answer prompt (EN) =====
ANSWER_PROMPT = PromptTemplate.from_template("""
You are the Team 3 answer generator in a RAG pipeline.

You will receive:
- A refined English question
- The selected RAG query
- Retrieved document snippets as a concatenated preview (may be truncated)

Your job:
- Produce the final answer strictly in the requested output format and language.

Requested format: {out_type}
Requested language: {answer_language}

Format rules:
- qa: Start with a one-sentence direct answer. Then add 2–4 short bullets of evidence/caveats.
      If a procedure is requested, include a numbered list of steps.
- bulleted: 5–10 bullet points, one sentence each. One sub-bullet level allowed if essential.
- table: Markdown table. First row is header. Infer 3–7 sensible columns from the question context.
         Use "N/A" for missing values. Up to ~50 rows unless clearly requested otherwise.
- json: Return valid JSON ONLY (no code fences). Use concise, sensible keys; strings/numbers/arrays/objects only.
- report: Markdown H2 sections in order: "## Summary", "## Findings", "## Method", "## Limitations".
          Each section 2–5 sentences; include lists/tables inline if helpful.

Grounding rules:
- Use ONLY information present in the provided docs preview.
- If the docs do not contain enough information, respond with the following no-information message
  in the requested format and language:
  - Korean: "문서에 해당 정보가 없습니다."
  - English: "The documents do not contain that information."
- Do NOT invent or hallucinate facts.
- Do NOT include citations or URLs unless present in the docs preview.

Write STRICTLY in: {answer_language}

Inputs:
[Refined question]
{q_en_transformed}

[Selected RAG query]
{rag_query}

[Docs preview]
{docs_preview}

Answer:
""")

# ===== Agent =====
def agent_team3_answer_generation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate final answer using retrieved docs and output_format constraints.
       - Prefers web_docs over rag_docs.
       - Respects output_format: ["type","language"].
       - Returns state with 'generated_answer' set (string).
    """
    new_state = state.copy()

    question = new_state.get("q_en_transformed", "") or ""
    query = new_state.get("rag_query", "") or ""
    output_format = new_state.get("output_format", ["qa", "ko"]) or ["qa", "ko"]

    # Validate/normalize output_format
    allowed_types = {"report", "table", "bulleted", "json", "qa"}
    allowed_langs = {"ko", "en"}
    out_type = (output_format[0] if len(output_format) > 0 else "qa").strip().lower()
    answer_language = (output_format[1] if len(output_format) > 1 else "ko").strip().lower()
    if out_type not in allowed_types:
        out_type = "qa"
    if answer_language not in allowed_langs:
        answer_language = "ko"

    # Choose docs: prefer web_docs, fallback to rag_docs
    web_docs = new_state.get("web_docs", []) or []
    rag_docs = new_state.get("rag_docs", []) or []
    docs = web_docs if web_docs else rag_docs
    docs_preview = _docs_preview(docs)

    try:
        llm = _get_llm()
        chain = ANSWER_PROMPT | llm
        result = chain.invoke({
            "q_en_transformed": question,
            "rag_query": query,
            "docs_preview": docs_preview,
            "out_type": out_type,
            "answer_language": answer_language,
        })
        new_state["generated_answer"] = (result.content or "").strip()
        return new_state
    except Exception as e:
        print(f"❌ Team 3 Agent error: {e}")
        new_state["generated_answer"] = "답변 생성 실패" if answer_language == "ko" else "Answer generation failed"
        return new_state
