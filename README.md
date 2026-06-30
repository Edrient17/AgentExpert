# AgentExpert: LangGraph Multi-Agent RAG Q&A Platform

AgentExpert is a LangGraph-based multi-agent question-answering system. It analyzes a user question, retrieves supporting context from a local FAISS vector store and optional web research, then generates and evaluates a final answer.

## Quick Start

1. Create and activate a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Create a `.env` file in the project root.
4. Add PDF documents to a `data/` directory.
5. Run `scripts/ingest_data.py` to build the vector store.
6. Run `app.py` with Streamlit to use the chat interface.

```bash
python -m pip install -r requirements.txt
python scripts/ingest_data.py
streamlit run app.py
```

On Windows, one common virtual environment setup is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Required Environment Variables

```env
OPENAI_API_KEY=your_openai_api_key_here
TESSERACT_LANG=eng
```

`TESSERACT_LANG` is optional. The project defaults to English OCR (`eng`).

## Notes

- Tesseract is required only when OCR is needed during PDF ingestion.
- The local vector store is saved under `vector_store/`.
- The Streamlit app and MCP server both use the same LangGraph workflow.
