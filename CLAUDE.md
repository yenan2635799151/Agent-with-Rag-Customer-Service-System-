# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered vacuum cleaner customer service system. Backend: Python FastAPI + LangChain ReAct agent + Chroma vector DB + DeepSeek LLM. Frontend: Next.js (TypeScript).

## Running the Project

### Backend
```bash
# Start API server (port 8004)
python api.py
# or: uvicorn api:app --reload --port 8004

# Load knowledge base into vector store (run once or when data/ changes)
python -c "from rag.vector_store import VectorStoreService; vs = VectorStoreService(); vs.load_documents()"

# Test agent directly
python agent/react_agent.py

# Test vector store retrieval
python rag/vector_store.py

# Alternative Streamlit UI
streamlit run app.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev      # http://localhost:3000
npm run build
npm start
```

### Evaluation
```bash
python eval/eval_pipline.py
```

### Milvus (alternative vector store)
```bash
bash milvus.sh   # starts Docker container on localhost:19530
```

## Configuration

All config lives in `config/*.yml` — no `.env` file:
- `config/rag.yml` — DeepSeek API key/URL, embedding model path, retrieval k values
- `config/chroma.yml` — collection name, persist directory (`choma_db/`), chunk size, data path
- `config/agent.yml` — Amap weather API key, external CSV data path
- `config/prompts.yml` — paths to prompt template files

## Architecture

### Request Flow
```
POST /api/chat → ReactAgent.execute_stream() → LangChain agent loop
  ├── middleware: monitor_tool, log_before_model, report_prompt_switch
  ├── 7 tools: rag_summarize, get_weather, get_user_location, get_user_id,
  │            get_current_month, fetch_external_data, fill_context_for_report
  └── DeepSeek LLM (streaming SSE → frontend)
```

### RAG Pipeline (`rag/`)
`VectorStoreService.get_retriever()` returns a `HybridRetriever` that chains:
1. Dense retrieval via Chroma (top `vector_k=30`)
2. BM25 reranking (top `bm25_k=10`)
3. Cross-encoder reranking via `Reranker` — gracefully skipped if model unavailable (top `final_k=3`)

Knowledge base documents (`data/*.txt`, `data/*.pdf`) are ingested with MD5 dedup tracked in `md5.txt`.

### Dynamic Prompt Switching
The `fill_context_for_report` tool sets `runtime.context["report"] = True` (via `monitor_tool` middleware). The `report_prompt_switch` `@dynamic_prompt` middleware then switches the system prompt to `prompts/report_prompt.txt` for all subsequent model calls in that request.

### Frontend Streaming
`frontend/src/hooks/useChat.ts` consumes SSE from `/api/chat`. `next.config.ts` proxies `/api/*` to `http://localhost:8004`.

### Vector Store Options
- **Chroma** (default): `rag/vector_store.py`, persists to `choma_db/`
- **Milvus** (alternative): `rag/m_vector_store.py`, requires Docker

## Key Files

| File | Role |
|------|------|
| `api.py` | FastAPI entry point, IP extraction, SSE streaming |
| `agent/react_agent.py` | Agent initialization with tools + middleware |
| `agent/tools/agent_tools.py` | All 7 tool definitions |
| `agent/tools/middleware.py` | Tool monitoring, logging, dynamic prompt switch |
| `rag/vector_store.py` | Chroma store, document ingestion, hybrid retriever |
| `rag/rag_service.py` | RAG chain (retriever + prompt + LLM) |
| `model/factory.py` | Initializes `chat_model` (DeepSeek) and `embed_model` (local BGE) |
| `utils/ip_context.py` | `ContextVar` for passing client IP through async context |
| `prompts/main_prompt.txt` | Main ReAct system prompt (Chinese) |
| `prompts/report_prompt.txt` | Report generation prompt, activated by middleware |
