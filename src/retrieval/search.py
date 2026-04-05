"""
Semantic search — chunk-level and page-level indexes.

Usage:
    python search.py                              # interactive REPL
    python search.py "what is RAG?"               # chunk search (default)
    python search.py "what is RAG?" --mode page   # page search
    python search.py "what is RAG?" --mode both   # chunk + page side-by-side
    python search.py "what is RAG?" --top 5       # top 5 results
    python search.py "what is RAG?" --file rag_explained.txt  # filter by file
"""

import sys, argparse, chromadb
from typing import List, Optional
from chromadb import EmbeddingFunction, Embeddings

# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_PATH      = "./chroma_db"
COLLECTION       = "documents"
PAGE_COLLECTION  = "page_index"
HF_MODEL         = "BAAI/bge-small-en-v1.5"
DEFAULT_TOP      = 3

# ── Embedder ───────────────────────────────────────────────────────────────────
class HuggingFaceEmbedder(EmbeddingFunction):
    def __init__(self, model_name: str = HF_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")
        print(f"  Loading model '{model_name}' ...")
        self.model = SentenceTransformer(model_name)
        print(f"  ✓ Model ready  (dim={self.model.get_sentence_embedding_dimension()})\n")

    def __call__(self, input: List[str]) -> Embeddings:
        return self.model.encode(
            input, show_progress_bar=False, normalize_embeddings=True
        ).tolist()

# ── Search ─────────────────────────────────────────────────────────────────────
def search(
    query     : str,
    collection,
    top_k     : int           = DEFAULT_TOP,
    filename  : Optional[str] = None,
    min_score : float         = 0.0,
    mode      : str           = "chunk",   # "chunk" | "page"
) -> List[dict]:
    where   = {"filename": filename} if filename else None
    results = collection.query(
        query_texts = [query],
        n_results   = top_k,
        where       = where,
        include     = ["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = round(1 - dist, 4)
        if score < min_score:
            continue

        if mode == "page":
            location = f"page {meta['page_number']}/{meta['page_total']}"
        else:
            location = f"chunk {meta['chunk_index'] + 1}/{meta['chunk_total']}"

        hits.append({
            "rank"      : len(hits) + 1,
            "score"     : score,
            "filename"  : meta["filename"],
            "location"  : location,
            "word_count": meta["word_count"],
            "text"      : doc,
        })
    return hits

# ── Display ────────────────────────────────────────────────────────────────────
def print_results(query: str, hits: List[dict], mode: str) -> None:
    label = "PAGE INDEX" if mode == "page" else "CHUNK INDEX"
    print(f"\n{'─' * 64}")
    print(f"  🔍 {label}  |  Query: {query}  |  {len(hits)} hit(s)")
    print(f"{'─' * 64}")
    if not hits:
        print("  No results found.")
        return
    for h in hits:
        bar = "█" * int(h["score"] * 20)
        print(f"\n  #{h['rank']}  [{h['score']:.4f}] {bar}")
        print(f"  📄 {h['filename']}  ·  {h['location']}  ·  {h['word_count']} words")
        preview = h["text"][:300].strip()
        print(f"  {preview}{'...' if len(h['text']) > 300 else ''}")
    print()

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Semantic search — chunk + page indexes")
    parser.add_argument("query",   nargs="?", default=None)
    parser.add_argument("--mode",  choices=["chunk", "page", "both"], default="chunk",
                        help="Search chunk index, page index, or both (default: chunk)")
    parser.add_argument("--top",   type=int,   default=DEFAULT_TOP)
    parser.add_argument("--file",  type=str,   default=None,  help="Filter to a specific filename")
    parser.add_argument("--min",   type=float, default=0.0,   help="Minimum similarity score 0–1")
    args = parser.parse_args()

    embedder    = HuggingFaceEmbedder(HF_MODEL)
    client      = chromadb.PersistentClient(path=CHROMA_PATH)
    chunk_col   = client.get_collection(COLLECTION,      embedding_function=embedder)
    page_col    = client.get_collection(PAGE_COLLECTION, embedding_function=embedder)

    print(f"  Chunk index : {chunk_col.count()} chunks")
    print(f"  Page index  : {page_col.count()} pages\n")

    def run_query(query: str) -> None:
        if args.mode in ("chunk", "both"):
            hits = search(query, chunk_col, args.top, args.file, args.min, mode="chunk")
            print_results(query, hits, mode="chunk")
        if args.mode in ("page", "both"):
            hits = search(query, page_col, args.top, args.file, args.min, mode="page")
            print_results(query, hits, mode="page")

    # Single query
    if args.query:
        run_query(args.query)
        return

    # Interactive REPL
    print("  Semantic Search REPL")
    print("  Commands: mode chunk|page|both  ·  top <n>  ·  file <name>  ·  min <score>  ·  clear  ·  exit")
    while True:
        try:
            query = input("\n  🔍 Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!"); break

        if not query: continue
        if query.lower() in {"exit", "quit", "q"}: print("  Bye!"); break

        # Inline commands
        if query.lower().startswith("mode "):
            v = query.split()[1]
            if v in ("chunk", "page", "both"): args.mode = v; print(f"  mode → {args.mode}")
            else: print("  mode must be: chunk | page | both")
            continue
        if query.lower().startswith("top "):
            args.top = int(query.split()[1]); print(f"  top_k → {args.top}"); continue
        if query.lower().startswith("file "):
            args.file = query.split(None, 1)[1]; print(f"  filter → {args.file}"); continue
        if query.lower().startswith("min "):
            args.min = float(query.split()[1]); print(f"  min_score → {args.min}"); continue
        if query.lower() == "clear":
            args.file = None; args.min = 0.0; print("  Filters cleared."); continue

        run_query(query)

if __name__ == "__main__":
    main()
