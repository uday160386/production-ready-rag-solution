"""
Cache-Augmented Generation (CAG) using Redis + ChromaDB + Google Gemini.

Architecture:
    Query → Redis cache hit?  ──yes──► return cached answer
                 │ no
                 ▼
           ChromaDB semantic search (retrieve top-k chunks)
                 │
                 ▼
           Gemini (generate answer from chunks)
                 │
                 ▼
           Store in Redis (TTL) → return answer

Setup:
    pip install google-genai redis chromadb sentence-transformers

    # Start Redis:
    docker run -d -p 6379:6379 redis
    # or: brew install redis && redis-server

    # Set your Google API key:
    export GOOGLE_API_KEY="..."

Usage:
    python rag.py                          # interactive REPL
    python rag.py "what is RAG?"           # single query
    python rag.py "what is RAG?" --top 5   # top 5 chunks as context
    python rag.py --flush                  # clear all cached answers
    python rag.py --stats                  # show cache stats
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google")
import os, sys, hashlib, json, argparse, textwrap, chromadb
from typing import List, Optional
from chromadb import EmbeddingFunction, Embeddings
from context_engine import ContextEngine
from guardrails import Guardrails, GuardrailConfig, PipelineGuardrailResult

# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_PATH   = "./chroma_db"
COLLECTION      = "documents"
PAGE_COLLECTION = "page_index"
HF_MODEL      = "BAAI/bge-small-en-v1.5"
DEFAULT_TOP_K = 3
CACHE_TTL     = 3600            # seconds — 1 hour (None = never expire)
CACHE_PREFIX  = "rag:cache:"
REDIS_HOST    = "localhost"
REDIS_PORT    = 6379
REDIS_DB      = 0

# ── LLM provider ──────────────────────────────────────────────────────────────
# "gemini"  → Google Gemini (requires GOOGLE_API_KEY, has free quota limits)
# "ollama"  → Local Ollama  (free, no quota — install: https://ollama.com)
LLM_PROVIDER    = "ollama"

# ── Gemini config ──────────────────────────────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL    = "gemini-2.0-flash-lite"
#                 "gemini-1.5-pro"   # higher quality, larger context

# ── Ollama config ──────────────────────────────────────────────────────────────
OLLAMA_MODEL    = "llama3.2"    # pull first: ollama pull llama3.2
#                 "mistral"     # ollama pull mistral
#                 "phi3"        # ollama pull phi3  (lightweight)
#                 "gemma2"      # ollama pull gemma2

MAX_TOKENS          = 1024
MAX_CONTEXT_TOKENS  = 3000   # token budget for retrieved context
MMR_LAMBDA          = 0.7    # 1.0=pure relevance, 0.0=pure diversity
COMPRESS_RATIO      = 0.7    # fraction of sentences to keep per chunk

# ── Guardrail config ───────────────────────────────────────────────────────────
GUARDRAILS_ENABLED    = True
TOPIC_WHITELIST       = []     # e.g. ["AI", "python", "databases"] — empty = allow all
TOPIC_BLOCK_OFF_TOPIC = False  # True = hard block, False = warn only
PII_REDACT            = True   # redact PII from queries and answers
RATE_LIMIT_ENABLED    = False  # enable Redis-backed rate limiting
MIN_CONFIDENCE_SCORE  = 0.3    # warn when top retrieval score is below this

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided "
    "context chunks. If the answer cannot be found in the context, say "
    "'I don't have enough information to answer that.' Be concise and accurate."
)

# ── HuggingFace Embedder ───────────────────────────────────────────────────────
class HuggingFaceEmbedder(EmbeddingFunction):
    def __init__(self, model_name: str = HF_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")
        print(f"  Loading embedding model '{model_name}' ...")
        self.model = SentenceTransformer(model_name)
        print(f"  ✓ Embedder ready  (dim={self.model.get_sentence_embedding_dimension()})")

    def __call__(self, input: List[str]) -> Embeddings:
        return self.model.encode(
            input, show_progress_bar=False, normalize_embeddings=True
        ).tolist()

# ── Redis Cache ────────────────────────────────────────────────────────────────
class RedisCache:
    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT, db: int = REDIS_DB):
        try:
            import redis
        except ImportError:
            raise ImportError("Run: pip install redis")
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.client.ping()
        print(f"  ✓ Redis connected  ({host}:{port})")

    def _key(self, query: str, top_k: int) -> str:
        raw = json.dumps({"q": query.strip().lower(), "k": top_k}, sort_keys=True)
        h   = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"{CACHE_PREFIX}{h}"

    def get(self, query: str, top_k: int) -> Optional[dict]:
        data = self.client.get(self._key(query, top_k))
        return json.loads(data) if data else None

    def set(self, query: str, top_k: int, payload: dict) -> None:
        self.client.set(self._key(query, top_k), json.dumps(payload), ex=CACHE_TTL)

    def flush(self) -> int:
        keys = self.client.keys(f"{CACHE_PREFIX}*")
        if keys:
            self.client.delete(*keys)
        return len(keys)

    def stats(self) -> dict:
        keys = self.client.keys(f"{CACHE_PREFIX}*")
        return {"cached_queries": len(keys), "ttl_seconds": CACHE_TTL}

# ── Retrieval ──────────────────────────────────────────────────────────────────
def _query_collection(collection, query: str, top_k: int) -> List[dict]:
    results = collection.query(
        query_texts = [query],
        n_results   = top_k,
        include     = ["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({"text": doc, "meta": meta, "score": round(1 - dist, 4)})
    return hits

def retrieve(query: str, chunk_col, page_col, top_k: int) -> List[dict]:
    """
    Two-stage retrieval:
    1. Page index  — find the most relevant pages (broad context)
    2. Chunk index — find the most relevant fine-grained chunks
    Merges both, deduplicates by filename+location, re-ranks by score.
    """
    page_hits  = _query_collection(page_col,  query, top_k)
    chunk_hits = _query_collection(chunk_col, query, top_k)

    seen, merged = set(), []

    for h in page_hits:
        key = f"{h['meta']['filename']}:page{h['meta']['page_number']}"
        if key not in seen:
            seen.add(key)
            merged.append({
                "text"    : h["text"],
                "filename": h["meta"]["filename"],
                "location": f"page {h['meta']['page_number']}/{h['meta']['page_total']}",
                "score"   : h["score"],
                "source"  : "page",
            })

    for h in chunk_hits:
        key = f"{h['meta']['filename']}:chunk{h['meta']['chunk_index']}"
        if key not in seen:
            seen.add(key)
            merged.append({
                "text"    : h["text"],
                "filename": h["meta"]["filename"],
                "location": f"chunk {h['meta']['chunk_index'] + 1}/{h['meta']['chunk_total']}",
                "score"   : h["score"],
                "source"  : "chunk",
            })

    # Re-rank by score descending, keep top_k * 2 for richer context
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[: top_k * 2]

# ── Context Engine (module-level singleton) ───────────────────────────────────
_context_engine: Optional[ContextEngine] = None

def get_context_engine() -> ContextEngine:
    global _context_engine
    if _context_engine is None:
        _context_engine = ContextEngine(
            max_context_tokens = MAX_CONTEXT_TOKENS,
            mmr_lambda         = MMR_LAMBDA,
            compress_ratio     = COMPRESS_RATIO,
        )
    return _context_engine

# ── Guardrails (module-level singleton) ───────────────────────────────────────
_guardrails: Optional[Guardrails] = None

def get_guardrails() -> Optional[Guardrails]:
    global _guardrails
    if not GUARDRAILS_ENABLED:
        return None
    if _guardrails is None:
        _guardrails = Guardrails(GuardrailConfig(
            topic_whitelist       = TOPIC_WHITELIST,
            topic_block_off_topic = TOPIC_BLOCK_OFF_TOPIC,
            pii_redact            = PII_REDACT,
            rate_limit_enabled    = RATE_LIMIT_ENABLED,
            min_confidence_score  = MIN_CONFIDENCE_SCORE,
        ))
    return _guardrails

# ── Generation ────────────────────────────────────────────────────────────────
def _build_context(chunks: List[dict]) -> str:
    return "\n\n".join(
        f"[Source: {c['filename']} {c['location']} ({c['source']}) | score: {c['score']}]\n{c['text']}"
        for c in chunks
    )

def _generate_gemini(user_message: str) -> str:
    """Generate with Google Gemini + exponential backoff on 429."""
    try:
        from google import genai
        from google.genai import types
        from google.genai.errors import ClientError
    except ImportError:
        raise ImportError("Run: pip install google-genai")

    api_key = GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set.\n"
            "  Get a free key: https://aistudio.google.com/app/apikey\n"
            "  Then: export GOOGLE_API_KEY='...'"
        )

    client      = genai.Client(api_key=api_key)
    max_retries = 4
    wait        = 35
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model    = GEMINI_MODEL,
                contents = user_message,
                config   = types.GenerateContentConfig(
                    system_instruction = SYSTEM_PROMPT,
                    max_output_tokens  = MAX_TOKENS,
                ),
            )
            return response.text
        except ClientError as e:
            is_429 = (
                getattr(e, "status_code", None) == 429 or
                getattr(e, "code",        None) == 429 or
                "429" in str(e)
            )
            if is_429 and attempt < max_retries:
                import time
                print(f"  ⚠ Gemini quota hit. Waiting {wait}s... (retry {attempt}/{max_retries-1})")
                time.sleep(wait)
                wait *= 2
            else:
                raise

def _generate_ollama(messages: List[dict]) -> str:
    """Generate with local Ollama — accepts full messages list for multi-turn."""
    try:
        import ollama
    except ImportError:
        raise ImportError("Run: pip install ollama")
    response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    return response["message"]["content"]

def generate(query: str, chunks: List[dict], ctx: Optional[ContextEngine] = None) -> str:
    """Generate answer using context-engineered prompt."""
    if ctx is not None:
        prepared     = ctx.prepare(query, chunks)
        messages     = prepared["messages"]
        # For Gemini, flatten messages to a single string (Gemini uses system_instruction separately)
        user_message = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    else:
        # Fallback: plain context block (no context engineering)
        user_message = "Context:\n" + "\n\n".join(
            f"[Source: {c['filename']} {c['location']}]\n{c['text']}" for c in chunks
        ) + f"\n\nQuestion: {query}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

    if LLM_PROVIDER == "gemini":
        return _generate_gemini(user_message)
    elif LLM_PROVIDER == "ollama":
        return _generate_ollama(messages)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Choose 'gemini' or 'ollama'.")

# ── Full CAG pipeline ──────────────────────────────────────────────────────────
def rag_query(
    query      : str,
    chunk_col,
    page_col,
    cache      : RedisCache,
    top_k      : int  = DEFAULT_TOP_K,
    no_cache   : bool = False,
) -> dict:
    # 0. Input guardrails
    guards = get_guardrails()
    if guards:
        gr = guards.check_input(query)
        if not gr.passed:
            return {
                "query"    : query,
                "answer"   : f"⚠ Request blocked: {gr.block_reason}",
                "blocked"  : True,
                "blocked_by": gr.blocked_by,
                "sources"  : [],
                "cache_hit": False,
            }
        if gr.redacted_input:
            query = gr.redacted_input   # use sanitised query

    # 1. Cache lookup
    if not no_cache:
        cached = cache.get(query, top_k)
        if cached:
            cached["cache_hit"] = True
            return cached

    # 2. Retrieve from ChromaDB (two-stage: page + chunk)
    chunks = retrieve(query, chunk_col, page_col, top_k)

    # 3. Context engineering — rewrite, MMR, compress, budget, build prompt
    ctx    = get_context_engine()
    prepared = ctx.prepare(query, chunks, top_k=top_k)

    # 4. Generate
    answer = generate(prepared["rewritten_query"], prepared["chunks"], ctx=ctx)

    # 5. Record turn in conversation memory
    ctx.record_answer(query, answer)

    # 4. Output guardrails
    if guards:
        og = guards.check_output(answer, chunks)
        if og.redacted_output:
            answer = og.redacted_output
        guardrail_warnings = og.warnings
    else:
        guardrail_warnings = []

    # 5. Cache & return
    payload = {
        "query"    : query,
        "answer"   : answer,
        "model"    : OLLAMA_MODEL if LLM_PROVIDER == "ollama" else GEMINI_MODEL,
        "sources"  : [{"file": c["filename"], "location": c["location"], "source": c["source"], "score": c["score"]} for c in chunks],
        "cache_hit"          : False,
        "context_steps"      : prepared.get("steps", {}),
        "guardrail_warnings" : guardrail_warnings,
    }
    if not no_cache:
        cache.set(query, top_k, payload)

    return payload

# ── Pretty print ───────────────────────────────────────────────────────────────
def print_result(result: dict) -> None:
    hit   = result.get("cache_hit", False)
    label = "⚡ CACHE HIT" if hit else f"🔍 RETRIEVED + GENERATED  [{result.get('model', GEMINI_MODEL)}]"
    print(f"\n{'─' * 64}")
    print(f"  {label}")
    print(f"{'─' * 64}")
    print(f"\n  Q: {result['query']}\n")
    for line in textwrap.wrap(result["answer"], width=62):
        print(f"  {line}")
    print(f"\n  Sources:")
    for s in result["sources"]:
        print(f"    • {s['file']}  {s['location']}  [{s['source']}]  (score: {s['score']})")
    print()

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Cache-Augmented RAG with Redis + ChromaDB + Google Gemini"
    )
    parser.add_argument("query",      nargs="?", default=None,        help="Query (omit for interactive REPL)")
    parser.add_argument("--top",      type=int,  default=DEFAULT_TOP_K, help=f"Top-k chunks (default: {DEFAULT_TOP_K})")
    parser.add_argument("--no-cache", action="store_true",            help="Bypass cache for this query")
    parser.add_argument("--flush",    action="store_true",            help="Flush all cached answers and exit")
    parser.add_argument("--stats",    action="store_true",            help="Show cache stats and exit")
    args = parser.parse_args()

    model_label = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else GEMINI_MODEL
    print(f"\n  Cache-Augmented RAG  (Redis + ChromaDB + {LLM_PROVIDER.title()} [{model_label}])")
    print("  " + "─" * 55)

    # Connect Redis
    try:
        cache = RedisCache()
    except Exception as e:
        print(f"\n  ✗ Redis not available: {e}")
        print("    Start Redis:  docker run -d -p 6379:6379 redis")
        print("              or: brew install redis && redis-server\n")
        sys.exit(1)

    if args.flush:
        n = cache.flush()
        print(f"\n  ✓ Flushed {n} cached entr{'y' if n == 1 else 'ies'} from Redis.\n")
        return

    if args.stats:
        print(f"\n  Cache stats: {cache.stats()}\n")
        return

    # Connect ChromaDB
    embedder   = HuggingFaceEmbedder(HF_MODEL)
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    chunk_col  = client.get_or_create_collection(COLLECTION, embedding_function=embedder, metadata={"hnsw:space": "cosine"})
    try:
        page_col = client.get_collection(PAGE_COLLECTION, embedding_function=embedder)
        print(f"  ✓ Chunk index '{COLLECTION}'  — {chunk_col.count()} chunks")
        print(f"  ✓ Page index  '{PAGE_COLLECTION}' — {page_col.count()} pages\n")
    except Exception:
        print(f"  ⚠ Page index not found — falling back to chunk index only")
        print(f"    Re-run create_vector_db.py to build the page index.\n")
        page_col = chunk_col

    # Single query mode
    if args.query:
        result = rag_query(args.query, chunk_col, page_col, cache, top_k=args.top, no_cache=args.no_cache)
        print_result(result)
        return

    # Interactive REPL
    print("  Type your question (or 'flush', 'stats', 'exit')")
    while True:
        try:
            query = input("\n  💬 Ask: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!"); break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("  Bye!"); break
        if query.lower() == "flush":
            n = cache.flush(); print(f"  ✓ Flushed {n} cached entries."); continue
        if query.lower() == "stats":
            print(f"  {cache.stats()}"); continue

        try:
            result = rag_query(query, chunk_col, page_col, cache, top_k=args.top, no_cache=args.no_cache)
            print_result(result)
        except Exception as e:
            print(f"\n  ✗ Error: {e}\n")

if __name__ == "__main__":
    main()
