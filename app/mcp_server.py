import os, sys, json, asyncio, traceback

# ── Save real stdout BEFORE anything touches it ────────────────────────────────
_stdout_fd = os.dup(1)
os.dup2(2, 1)  # fd1 → stderr so stray prints never corrupt JSON-RPC

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

def log(msg):
    print(f"[mcp] {msg}", file=sys.stderr, flush=True)

log(f"dir={_HERE}")

# ── Imports ────────────────────────────────────────────────────────────────────
try:
    from mcp.server.fastmcp import FastMCP
    log("✓ fastmcp")
except Exception as e:
    log(f"✗ fastmcp: {e}\n{traceback.format_exc()}")
    sys.exit(1)

try:
    import warnings; warnings.filterwarnings("ignore", category=FutureWarning)
    import chromadb
    log("✓ chromadb")
except Exception as e:
    log(f"✗ chromadb: {e}"); sys.exit(1)

try:
    from rag import (
        HuggingFaceEmbedder, RedisCache, retrieve, generate, rag_query,
        get_context_engine, COLLECTION, PAGE_COLLECTION, HF_MODEL, DEFAULT_TOP_K,
    )
    log("✓ rag")
except Exception as e:
    log(f"✗ rag: {e}\n{traceback.format_exc()}"); sys.exit(1)

# ── Lazy resources ─────────────────────────────────────────────────────────────
_r: dict = {}

def R() -> dict:
    if _r: return _r
    log("init resources...")
    emb = HuggingFaceEmbedder(HF_MODEL); log("✓ embedder")
    db  = chromadb.PersistentClient(path=os.path.join(_HERE, "chroma_db"))
    cc  = db.get_or_create_collection(COLLECTION, embedding_function=emb,
                                       metadata={"hnsw:space": "cosine"})
    log(f"✓ chunks={cc.count()}")
    try:
        pc = db.get_collection(PAGE_COLLECTION, embedding_function=emb)
        log(f"✓ pages={pc.count()}")
    except Exception:
        pc = cc; log("⚠ no page_index")
    try:
        ca = RedisCache(); log("✓ redis")
    except Exception:
        ca = None; log("⚠ no redis")
    _r.update({"cc": cc, "pc": pc, "ca": ca})
    return _r

# ── FastMCP tools ──────────────────────────────────────────────────────────────
mcp = FastMCP("rag")

@mcp.tool()
def rag_query_tool(query: str, top_k: int = DEFAULT_TOP_K, no_cache: bool = False) -> str:
    """Answer a question using RAG (ChromaDB retrieval + LLM generation + Redis cache)."""
    r = R()
    if r["ca"]:
        result = rag_query(query=query, chunk_col=r["cc"], page_col=r["pc"],
                           cache=r["ca"], top_k=top_k, no_cache=no_cache)
    else:
        chunks = retrieve(query, r["cc"], r["pc"], top_k)
        answer = generate(query, chunks)
        result = {"query": query, "answer": answer, "cache_hit": False,
                  "sources": [{"file": c["filename"], "location": c["location"],
                                "score": c["score"]} for c in chunks]}
    return json.dumps(result, indent=2)

@mcp.tool()
def semantic_search(query: str, top_k: int = DEFAULT_TOP_K,
                    mode: str = "both", filename: str = "") -> str:
    """Search indexed documents and return matching chunks without LLM generation."""
    r     = R()
    where = {"filename": filename} if filename else None
    out   = {}
    if mode in ("chunk", "both"):
        res = r["cc"].query(query_texts=[query], n_results=top_k, where=where,
                            include=["documents","metadatas","distances"])
        out["chunk_results"] = [
            {"text": d, "filename": m["filename"],
             "location": f"chunk {m['chunk_index']+1}/{m['chunk_total']}",
             "score": round(1-dist, 4)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
    if mode in ("page", "both"):
        res = r["pc"].query(query_texts=[query], n_results=top_k, where=where,
                            include=["documents","metadatas","distances"])
        out["page_results"] = [
            {"text": d, "filename": m["filename"],
             "location": f"page {m['page_number']}/{m['page_total']}",
             "score": round(1-dist, 4)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]
    return json.dumps(out, indent=2)

@mcp.tool()
def list_documents() -> str:
    """List all documents currently indexed in ChromaDB."""
    r    = R()
    res  = r["cc"].get(include=["metadatas"])
    seen, docs = set(), []
    for m in res["metadatas"]:
        if m["filename"] not in seen:
            seen.add(m["filename"])
            docs.append({"filename": m["filename"], "ext": m["ext"],
                          "chunks": m["chunk_total"]})
    return json.dumps({"documents": docs, "total": len(docs)}, indent=2)

@mcp.tool()
def cache_stats() -> str:
    """Return Redis cache statistics."""
    r = R()
    return json.dumps(r["ca"].stats() if r["ca"] else {"error": "Redis unavailable"}, indent=2)

@mcp.tool()
def flush_cache() -> str:
    """Flush all cached RAG answers from Redis."""
    r = R()
    return json.dumps({"flushed": r["ca"].flush() if r["ca"] else 0})

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log("running...")
    # Restore real stdout for FastMCP's stdio transport
    os.dup2(_stdout_fd, 1)
    mcp.run(transport="stdio")
