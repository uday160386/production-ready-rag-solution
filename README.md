#  RAG [Standard & Agentic] Pipeline

> Cache-Augmented Generation with Context Engineering, Semantic Search, Embeddings, Chunking, page Index,Web Chat UI, and MCP Server for Claude Desktop.


##  Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/uday160386/production-ready-rag-solution.git
cd production-ready-rag-solution.

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Add Documents

Drop any `.txt`, `.md`, `.pdf`, `.json`, or `.csv` files into `data/`.

### 4. Ingest Documents

```bash
python src/ingestion/create_vector_db.py
```

### 5. Choose Your Interface

**Web Chat UI:**
```bash
python /server/chat_server.py
# Open http://localhost:8000
```

**CLI Search:**
```bash
python src/retrieval/search.py                         # interactive REPL
python src/retrieval/search.py "what is RAG?"          # single query
python src/retrieval/search.py "query" --mode page     # page-level search
python src/retrieval/search.py "query" --mode both     # chunk + page
```

**RAG CLI:**
```bash
python src/generation/rag.py                           # interactive REPL
python src/generation/rag.py "what is RAG?"            # single query
python src/generation/rag.py --flush                   # clear cache
python src/generation/rag.py --stats                   # cache stats
```

** Claude Desktop (MCP):**
```bash
./server/mcp_server.py   # or configure via claude_desktop_config.json
```

## MCP Setup (Claude Desktop @ MacBook)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "rag": {
      "command": "/path/to/.venv/bin/python",
      "args": ["/path/to/src/scripts/mcp_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/src/generation",
        "GOOGLE_API_KEY": "your-key",
        "PATH": "/path/to/.venv/bin:/usr/local/bin:/usr/bin:/bin"
      }
    }
  }
}
```

**Available MCP Tools:**

| Tool | Description |
|---|---|
| `rag_query` | Full RAG pipeline — cache → retrieve → generate |
| `semantic_search` | Search only, no LLM |
| `list_documents` | List indexed files |
| `cache_stats` | Redis statistics |
| `flush_cache` | Clear cached answers |

## Configuration

Edit constants at the top of each file, or set via `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `gemini` |
| `OLLAMA_MODEL` | `llama3.2` | Local Ollama model |
| `GEMINI_MODEL` | `gemini-2.0-flash-lite` | Gemini model |
| `GOOGLE_API_KEY` | — | Required for Gemini |
| `HF_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `CHUNK_SIZE` | `200` | Words per chunk |
| `CHUNK_OVERLAP` | `40` | Overlap between chunks |
| `CACHE_TTL` | `3600` | Redis cache TTL (seconds) |
| `MAX_CONTEXT_TOKENS` | `3000` | Token budget for context |
| `MMR_LAMBDA` | `0.7` | Relevance vs diversity (0–1) |
| `COMPRESS_RATIO` | `0.7` | Sentence keep ratio |


## Embedding Models

| Model | Dims | Size | Best For |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | 130MB | Default — best free RAG model |
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | Better quality |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.3GB | Best quality |
| `all-MiniLM-L6-v2` | 384 | 90MB | Lightweight fallback |

## Dependencies

- **chromadb** — vector database
- **sentence-transformers** — embeddings hugging face
- **redis** — answer cache
- **fastapi + uvicorn** — web server
- **mcp** — Model Context Protocol server
- **google-genai** — Gemini API
- **ollama** — local LLM client
- **pypdf** — PDF text extraction
