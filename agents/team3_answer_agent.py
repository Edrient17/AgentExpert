# agents/team3_answer_agent.py
from typing import List, Union, Dict, Any
import os

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# ===== Lazy LLM =====
def _get_llm() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 또는 환경변수를 확인하세요.")
    return ChatOpenAI(model="o3")

# ===== Docs util =====
def _docs_preview(docs: List[Union[str, object]], max_chars: int = 10000) -> str:
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
- Retrieved passages as a concatenated preview (may be truncated)

Your job:
- Produce the final answer strictly in the requested output format and language.

<think>
Passages:
{passages}

I will use both the information from these passages and my own prior knowledge to answer the question.
First, I will carefully examine whether the passages contain any misinformation, contradictions, or irrelevant details.
Then, I will combine verified facts from the passages with only safe, generally accepted prior knowledge to derive the correct answer.
If conflicts remain or the evidence is insufficient, I will output the no-information message instead of guessing.
I will NOT reveal any reasoning or the content of this <think> block in the final output.
</think>

Requested format: {out_type}
Requested language: {answer_language}

Format guidelines (flexible within type):
- qa:
  • Begin with a one-sentence direct answer.  
  • Then provide ample explanation (multiple short paragraphs and/or 3–8 bullets).  
  • If a procedure is involved, include a numbered step list. May add "Edge cases", "Tips" sections if useful.
- bulleted:
  • 8–15 bullets with meaningful substance (≤40 words each).  
  • You may group bullets by mini-headings (bold) and use one level of sub-bullets when needed.
- table:
  • Produce a Markdown table with a header row; derive 3–9 sensible columns from the question/context.  
  • Include as many rows as needed (up to ~100). Add a short "Notes" paragraph below if clarification helps.  
  • Use "N/A" for missing values.
- json:
  • Return valid JSON ONLY (no code fences).  
  • Always include an "answer" string. You may add keys like "evidence", "steps", "assumptions", "limitations", "confidence" (0–1), etc.  
  • Keep keys concise; use arrays/objects as needed.
- report:
  • Design your own section plan (3–10 H2 sections) tailored to the question.
  • Each section 3–8 sentences; include lists/tables where helpful. Subsections (H3) allowed.

Grounding & safety rules:
- Use the passages as primary evidence. You MAY use prior knowledge only if it does not contradict the passages.
- If the passages are insufficient or conflicting, respond with the no-information message in the requested language and format:
  - ko: "문서에 해당 정보가 없습니다."
  - en: "The documents do not contain that information."
- Do NOT invent or hallucinate facts.
- Do NOT include citations or URLs unless they explicitly appear in the passages.
- DO NOT reveal the content of <think> or any reasoning steps.

Write STRICTLY in: {answer_language}

Inputs:
[Refined question]
{q_en_transformed}

Answer:
""")

# ===== Agent =====
def agent_team3_answer_generation(state: Dict[str, Any]) -> Dict[str, Any]:
    
    new_state = state.copy()

    question = new_state.get("q_en_transformed", "") or ""
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

    # Choose docs
    web_docs = new_state.get("web_docs", []) or []
    rag_docs = new_state.get("rag_docs", []) or []
    docs = web_docs if web_docs else rag_docs
    passages = _docs_preview(docs)

    try:
        llm = _get_llm()
        chain = ANSWER_PROMPT | llm
        result = chain.invoke({
            "q_en_transformed": question,
            "passages": passages,
            "out_type": out_type,
            "answer_language": answer_language,
        })
        new_state["generated_answer"] = (result.content or "").strip()
        return new_state
    except Exception as e:
        print(f"❌ Team 3 Agent error: {e}")
        new_state["generated_answer"] = "답변 생성 실패" if answer_language == "ko" else "Answer generation failed"
        return new_state
