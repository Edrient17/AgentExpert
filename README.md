# AgentExpert

AgentExpert is a LangGraph-based multi-agent RAG Q&A platform. It analyzes a user question, retrieves supporting context from a local FAISS vector store and optional web research, then generates and evaluates a final answer.

## Architecture

```text
Supervisor
  -> Query Team
       -> Query Worker
       -> Query Evaluator
  -> Retrieval Team
       -> RAG Worker
       -> Web Worker
       -> Document Evaluator
  -> Answer Team
       -> Answer Generator
       -> Answer Evaluator
```

The implementation is organized under `src/`:

```text
src/
  agents/      Role-specific agent nodes.
  graph/       LangGraph team graphs and supervisor graph.
  prompts/     YAML prompt registry.
  schema/      Shared state and structured output schemas.
  tools/       RAG, web research, OCR-adjacent helpers, and formatting tools.
scripts/       Standalone utilities such as PDF ingestion and naive RAG debugging.
```

## Setup

Create and activate a Python virtual environment, then install dependencies.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Set at least:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Environment Variables

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | Yes | N/A | OpenAI API key used by LLM and embedding calls. |
| `TESSERACT_LANG` | No | `eng` | OCR language for PDF ingestion. Use `kor+eng` if Korean language data is installed. |
| `TESSERACT_CMD` | No | auto-detected | Explicit path to `tesseract.exe` when auto-detection is not enough. |
| `TESSDATA_PREFIX` | No | auto-detected | Explicit path to the Tesseract `tessdata` directory. |
| `PORT` | No | `8080` | Port used by `api.py`. |
| `ENABLE_WEB_RESEARCH` | No | `false` | Enables the current LLM-based web research fallback when RAG context is insufficient. |

## Build the Vector Store

Place PDF files in a local `data/` directory, then run:

```bash
python scripts/ingest_data.py
```

The FAISS index is written to `vector_store/`.

Tesseract is only needed when PDF ingestion requires OCR. English OCR works with the default `TESSERACT_LANG=eng` if the English language pack is installed.

## Run the Streamlit App

```bash
streamlit run app.py
```

For a quick smoke test, ask:

```text
What is 2 plus 2? Answer briefly.
```

Simple questions skip retrieval and go directly from the Query Team to the Answer Team.

## Web Research Fallback

By default, web research is disabled:

```env
ENABLE_WEB_RESEARCH=false
```

The current web research tool is LLM-generated research summarization, not a live search API. Keep it disabled for stricter local-RAG behavior, or set `ENABLE_WEB_RESEARCH=true` if you explicitly want the Retrieval Team to use that fallback when local RAG does not return enough usable context.

## Run the MCP Server

```bash
python api.py
```

The MCP server exposes the same LangGraph workflow through the `ask_agent` tool.

## Run Tests

```bash
pytest
```

The current test suite avoids external API calls. It checks prompt loading, graph construction, answer context assembly, and document formatting helpers.

## Development Notes

- Prompts live in YAML files under `src/prompts/`.
- Agent routing is controlled by `src/graph/supervisor.py`.
- Shared workflow state is defined in `src/schema/state.py`.
- Structured LLM outputs are defined in `src/schema/outputs.py`.
- Local secrets should stay in `.env`; `.env.example` is safe to commit.
