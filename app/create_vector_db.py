import chromadb, os
from typing import List, Dict
from chromadb import EmbeddingFunction, Embeddings

# ── Config ─────────────────────────────────────────────────────────────────────
CHUNK_SIZE     = 200
CHUNK_OVERLAP  = 40
BATCH_SIZE     = 64
CHROMA_PATH    = "./chroma_db"
COLLECTION     = "documents"
PAGE_COLLECTION = "page_index"       # stores one doc per page for page-level search
HF_MODEL       = "BAAI/bge-small-en-v1.5"
#                "BAAI/bge-base-en-v1.5"
#                "BAAI/bge-large-en-v1.5"
#                "thenlper/gte-small"
#                "all-MiniLM-L6-v2"

# ── Chunking ───────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

# ── HuggingFace Embedder ───────────────────────────────────────────────────────
class HuggingFaceEmbedder(EmbeddingFunction):
    """sentence-transformers embedder — models cached locally after first download."""
    def __init__(self, model_name: str = HF_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")
        print(f"  Loading model '{model_name}' ...")
        self.model = SentenceTransformer(model_name)
        print(f"  ✓ Model ready  (dim={self.model.get_sentence_embedding_dimension()})")

    def __call__(self, input: List[str]) -> Embeddings:
        return self.model.encode(
            input, show_progress_bar=False, normalize_embeddings=True
        ).tolist()

# ── Page index builder ─────────────────────────────────────────────────────────
def extract_pages(filepath: str, ext: str) -> List[Dict]:
    """
    Extract content page-by-page.
    - PDFs  → one entry per PDF page  (page_number = real page)
    - Text  → split into ~500-word virtual pages
    Returns list of {page_number, content}
    """
    pages = []

    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(filepath)
            for i, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append({"page_number": i + 1, "content": text})
        except ImportError:
            print("  ⚠ pypdf not installed — run: pip install pypdf")
    else:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        # Split into virtual pages of ~500 words
        words     = content.split()
        PAGE_SIZE = 500
        for i in range(0, len(words), PAGE_SIZE):
            pages.append({
                "page_number": i // PAGE_SIZE + 1,
                "content"    : " ".join(words[i : i + PAGE_SIZE]),
            })

    return pages

# ── Load documents ─────────────────────────────────────────────────────────────
data_dir      = "./data"
supported_ext = {".txt", ".md", ".csv", ".json", ".pdf"}

# Storage for chunk-level index
all_chunks, all_ids, all_metadatas = [], [], []

# Storage for page-level index
page_docs, page_ids, page_metas = [], [], []

for filename in sorted(os.listdir(data_dir)):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in supported_ext:
        print(f"  ⚠ Skipping unsupported file: {filename}")
        continue

    filepath = os.path.join(data_dir, filename)
    base_id  = filename.replace(" ", "_").replace(".", "_")

    # ── Page index ─────────────────────────────────────────────────────────────
    pages = extract_pages(filepath, ext)
    if not pages:
        print(f"  ⚠ Skipping empty file: {filename}")
        continue

    for p in pages:
        page_docs.append(p["content"])
        page_ids.append(f"{base_id}_page{p['page_number']}")
        page_metas.append({
            "filename"   : filename,
            "path"       : filepath,
            "ext"        : ext,
            "page_number": p["page_number"],
            "page_total" : len(pages),
            "word_count" : len(p["content"].split()),
        })

    # ── Chunk index (fine-grained) ──────────────────────────────────────────────
    full_content = " ".join(p["content"] for p in pages)
    chunks = chunk_text(full_content)

    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_ids.append(f"{base_id}_chunk{i}")
        all_metadatas.append({
            "filename"   : filename,
            "path"       : filepath,
            "ext"        : ext,
            "chunk_index": i,
            "chunk_total": len(chunks),
            "page_total" : len(pages),
            "word_count" : len(chunk.split()),
        })

    print(f"  ✓ {filename} → {len(pages)} page(s), {len(chunks)} chunk(s)")

if not all_chunks:
    raise ValueError(f"No valid documents found in '{data_dir}'. Check the path and file types.")

print(f"\n  Chunks : {len(all_chunks)}")
print(f"  Pages  : {len(page_docs)}")

# ── Embed & index ──────────────────────────────────────────────────────────────
embedder = HuggingFaceEmbedder(model_name=HF_MODEL)
client   = chromadb.PersistentClient(path=CHROMA_PATH)

# Chunk-level collection
try: client.delete_collection(COLLECTION)
except Exception: pass

chunk_col = client.create_collection(
    COLLECTION,
    embedding_function=embedder,
    metadata={"hnsw:space": "cosine"},
)
for i in range(0, len(all_chunks), BATCH_SIZE):
    chunk_col.add(
        documents = all_chunks[i : i + BATCH_SIZE],
        ids       = all_ids[i : i + BATCH_SIZE],
        metadatas = all_metadatas[i : i + BATCH_SIZE],
    )
print(f"  ✓ Chunk index  → {chunk_col.count()} chunks  in '{COLLECTION}'")

# Page-level collection
try: client.delete_collection(PAGE_COLLECTION)
except Exception: pass

page_col = client.create_collection(
    PAGE_COLLECTION,
    embedding_function=embedder,
    metadata={"hnsw:space": "cosine"},
)
for i in range(0, len(page_docs), BATCH_SIZE):
    page_col.add(
        documents = page_docs[i : i + BATCH_SIZE],
        ids       = page_ids[i : i + BATCH_SIZE],
        metadatas = page_metas[i : i + BATCH_SIZE],
    )
print(f"  ✓ Page index   → {page_col.count()} pages   in '{PAGE_COLLECTION}'")

print(f"\n✅ Indexing complete  (model: {HF_MODEL})")
print(f"   Run search.py or rag.py to query your documents.")
