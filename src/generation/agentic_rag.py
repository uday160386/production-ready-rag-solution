"""
Agentic RAG — LLM-driven retrieval with tool use, self-reflection, and
iterative reasoning loops.

Architecture:
    Query
      │
      ▼
    Agent plans which tools to call
      │
      ├── search_chunks()    — semantic chunk search
      ├── search_pages()     — semantic page search
      ├── filter_by_file()   — retrieve from a specific document
      ├── summarise_doc()    — summarise an entire document
      └── answer()           — produce final answer with citations
      │
      ▼
    Agent reflects: "Is my answer complete?"
      ├── YES → return answer
      └── NO  → plan next tool call (up to MAX_ITERATIONS)

Usage:
    from agentic_rag import AgenticRAG
    agent = AgenticRAG(chunk_col, page_col, cache)
    result = agent.run("What are the key differences between HNSW and LSH?")
    print(result["answer"])
    print(result["trace"])   # full reasoning trace
"""

import json, re, sys
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_ITERATIONS   = 5       # max reasoning loops before forcing an answer
TOP_K_DEFAULT    = 4       # chunks per tool call
CONFIDENCE_THRESHOLD = 0.5 # min score to consider a chunk relevant

# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class ToolCall:
    name      : str
    args      : Dict[str, Any]
    result    : Any   = None
    error     : str   = ""

@dataclass
class AgentStep:
    iteration : int
    thought   : str
    tool_call : Optional[ToolCall] = None
    observation: str = ""

@dataclass
class AgentResult:
    query     : str
    answer    : str
    sources   : List[Dict]
    trace     : List[AgentStep]
    iterations: int
    cache_hit : bool = False

# ── Tool definitions ───────────────────────────────────────────────────────────
TOOLS = [
    {
        "name"       : "search_chunks",
        "description": "Search the chunk index for fine-grained relevant passages. Use for specific facts, definitions, or code snippets.",
        "parameters" : {
            "query": "search query string",
            "top_k": "number of results (default 4)",
        },
    },
    {
        "name"       : "search_pages",
        "description": "Search the page index for broader context. Use when you need a wider view of a topic or the question spans multiple sections.",
        "parameters" : {
            "query": "search query string",
            "top_k": "number of results (default 4)",
        },
    },
    {
        "name"       : "filter_by_file",
        "description": "Retrieve chunks from a specific document by filename. Use when you know which document contains the answer.",
        "parameters" : {
            "filename": "exact filename to filter on",
            "query"   : "search query within that file",
            "top_k"   : "number of results (default 4)",
        },
    },
    {
        "name"       : "summarise_doc",
        "description": "Get the full content of a specific document summarised. Use when you need a high-level overview.",
        "parameters" : {
            "filename": "exact filename to summarise",
        },
    },
    {
        "name"       : "answer",
        "description": "Produce the final answer to the user's question. Call this when you have enough context. Include inline source citations.",
        "parameters" : {
            "answer"  : "the complete answer with citations like [Source: filename]",
            "confident": "true if you found relevant information, false if unsure",
        },
    },
]

TOOLS_JSON = json.dumps(TOOLS, indent=2)

# ── Agent Prompt Templates ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an intelligent research agent with access to a document knowledge base.
Your job is to answer the user's question by strategically searching and reasoning over documents.

You have these tools available:
{tools}

Rules:
1. Think step by step. First analyse what the question needs, then choose the right tool.
2. You may call multiple tools across multiple iterations to build a complete answer.
3. Always cite your sources in the final answer using [Source: filename].
4. If search results are not relevant enough, try a different query or different tool.
5. Call `answer` when you have enough information — do NOT over-search.
6. Respond ONLY with valid JSON in this format:
   {{"thought": "your reasoning", "tool": "tool_name", "args": {{...}}}}
"""

ITERATION_PROMPT = """Previous steps:
{trace}

Retrieved so far:
{context}

User question: {query}

What is your next action? Respond with JSON only."""

FORCE_ANSWER_PROMPT = """You have reached the maximum iterations.
Based on everything retrieved so far, provide your best answer.

Context collected:
{context}

User question: {query}

Respond with:
{{"thought": "summarising findings", "tool": "answer", "args": {{"answer": "...", "confident": false}}}}"""

# ── Tool Executor ──────────────────────────────────────────────────────────────
class ToolExecutor:
    def __init__(self, chunk_col, page_col):
        self.chunk_col = chunk_col
        self.page_col  = page_col

    def execute(self, tool_call: ToolCall) -> str:
        try:
            if tool_call.name == "search_chunks":
                return self._search(self.chunk_col, tool_call.args, mode="chunk")
            elif tool_call.name == "search_pages":
                return self._search(self.page_col, tool_call.args, mode="page")
            elif tool_call.name == "filter_by_file":
                return self._filter_by_file(tool_call.args)
            elif tool_call.name == "summarise_doc":
                return self._summarise_doc(tool_call.args)
            elif tool_call.name == "answer":
                return "__ANSWER__"   # sentinel — agent is done
            else:
                return f"Unknown tool: {tool_call.name}"
        except Exception as e:
            return f"Tool error: {e}"

    def _search(self, col, args: dict, mode: str) -> str:
        query  = args.get("query", "")
        top_k  = int(args.get("top_k", TOP_K_DEFAULT))
        res    = col.query(
            query_texts = [query],
            n_results   = top_k,
            include     = ["documents", "metadatas", "distances"],
        )
        items = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            score = round(1 - dist, 4)
            if score < CONFIDENCE_THRESHOLD:
                continue
            loc = (f"page {meta['page_number']}/{meta['page_total']}"
                   if mode == "page"
                   else f"chunk {meta['chunk_index']+1}/{meta['chunk_total']}")
            items.append(
                f"[{meta['filename']} · {loc} · score {score}]\n{doc[:400]}"
            )
        return "\n\n".join(items) if items else "No relevant results found."

    def _filter_by_file(self, args: dict) -> str:
        filename = args.get("filename", "")
        query    = args.get("query", filename)
        top_k    = int(args.get("top_k", TOP_K_DEFAULT))
        try:
            res = self.chunk_col.query(
                query_texts = [query],
                n_results   = top_k,
                where       = {"filename": filename},
                include     = ["documents", "metadatas", "distances"],
            )
            items = []
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            ):
                score = round(1 - dist, 4)
                items.append(
                    f"[{meta['filename']} · chunk {meta['chunk_index']+1}/{meta['chunk_total']} · score {score}]\n{doc[:400]}"
                )
            return "\n\n".join(items) if items else f"No results in {filename}."
        except Exception as e:
            return f"Error filtering by file: {e}"

    def _summarise_doc(self, args: dict) -> str:
        filename = args.get("filename", "")
        try:
            res = self.chunk_col.get(
                where   = {"filename": filename},
                include = ["documents", "metadatas"],
            )
            if not res["documents"]:
                return f"No document found: {filename}"
            # Sort by chunk index and join
            pairs = sorted(
                zip(res["metadatas"], res["documents"]),
                key=lambda x: x[0].get("chunk_index", 0),
            )
            full = " ".join(doc for _, doc in pairs)
            # Return first 600 words as summary
            return " ".join(full.split()[:600]) + " [...]"
        except Exception as e:
            return f"Error summarising: {e}"

# ── LLM Caller ────────────────────────────────────────────────────────────────
def _call_llm(messages: List[Dict], llm_provider: str,
              ollama_model: str, gemini_model: str,
              google_api_key: str) -> str:
    """Call the configured LLM and return raw text."""
    if llm_provider == "ollama":
        try:
            import ollama
            response = ollama.chat(model=ollama_model, messages=messages)
            return response["message"]["content"]
        except ImportError:
            raise ImportError("Run: pip install ollama")

    elif llm_provider == "gemini":
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=google_api_key)
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user   = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            resp   = client.models.generate_content(
                model    = gemini_model,
                contents = user,
                config   = types.GenerateContentConfig(
                    system_instruction = system,
                    max_output_tokens  = 512,
                ),
            )
            return resp.text
        except ImportError:
            raise ImportError("Run: pip install google-genai")

    raise ValueError(f"Unknown LLM provider: {llm_provider}")

def _parse_tool_call(raw: str) -> Optional[ToolCall]:
    """Parse JSON tool call from LLM response, tolerating minor formatting issues."""
    try:
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        data = json.loads(raw)
        return ToolCall(
            name = data.get("tool", ""),
            args = data.get("args", {}),
        )
    except Exception:
        # Try extracting JSON object from response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return ToolCall(name=data.get("tool",""), args=data.get("args",{}))
            except Exception:
                pass
        return None

# ── Main Agent ────────────────────────────────────────────────────────────────
class AgenticRAG:
    """
    Agentic RAG with tool use, self-reflection, and iterative reasoning.

    The agent decides WHICH tools to call, WHEN to call them, and WHEN it has
    enough context to answer — unlike standard RAG which blindly retrieves top-k
    and passes everything to the LLM.
    """

    def __init__(
        self,
        chunk_col,
        page_col,
        cache           = None,
        llm_provider    : str = "ollama",
        ollama_model    : str = "llama3.2",
        gemini_model    : str = "gemini-2.0-flash-lite",
        google_api_key  : str = "",
        max_iterations  : int = MAX_ITERATIONS,
    ):
        self.executor       = ToolExecutor(chunk_col, page_col)
        self.cache          = cache
        self.llm_provider   = llm_provider
        self.ollama_model   = ollama_model
        self.gemini_model   = gemini_model
        self.google_api_key = google_api_key
        self.max_iterations = max_iterations

    def run(self, query: str, no_cache: bool = False) -> AgentResult:
        """Run the agentic RAG loop for a given query."""

        # 1. Cache check
        if self.cache and not no_cache:
            cached = self.cache.get(f"agent:{query}", 0)
            if cached:
                result = AgentResult(**cached)
                result.cache_hit = True
                return result

        trace     : List[AgentStep] = []
        collected : List[str]       = []   # accumulated context
        sources   : List[Dict]      = []   # for citation tracking

        system_msg = {
            "role"   : "system",
            "content": SYSTEM_PROMPT.format(tools=TOOLS_JSON),
        }

        for iteration in range(1, self.max_iterations + 1):

            # Build prompt
            trace_text   = self._format_trace(trace)
            context_text = "\n\n".join(collected[-6:]) if collected else "None yet."

            if iteration == self.max_iterations:
                # Force an answer on last iteration
                user_content = FORCE_ANSWER_PROMPT.format(
                    context=context_text, query=query
                )
            else:
                user_content = ITERATION_PROMPT.format(
                    trace=trace_text, context=context_text, query=query
                )

            messages = [
                system_msg,
                {"role": "user", "content": user_content},
            ]

            # 2. LLM decides next action
            try:
                raw = _call_llm(
                    messages       = messages,
                    llm_provider   = self.llm_provider,
                    ollama_model   = self.ollama_model,
                    gemini_model   = self.gemini_model,
                    google_api_key = self.google_api_key,
                )
            except Exception as e:
                step = AgentStep(iteration=iteration, thought="LLM error",
                                 observation=str(e))
                trace.append(step)
                break

            # 3. Parse tool call
            tool_call = _parse_tool_call(raw)
            if not tool_call:
                step = AgentStep(iteration=iteration, thought=raw[:200],
                                 observation="Could not parse tool call — retrying")
                trace.append(step)
                continue

            # Extract thought from raw response
            try:
                thought = json.loads(re.sub(r"```(?:json)?","",raw).strip()).get("thought","")
            except Exception:
                thought = raw[:200]

            # 4. Execute tool
            if tool_call.name == "answer":
                # Agent is done — extract final answer
                answer = tool_call.args.get("answer", "No answer provided.")
                step   = AgentStep(
                    iteration   = iteration,
                    thought     = thought,
                    tool_call   = tool_call,
                    observation = "Final answer produced.",
                )
                trace.append(step)

                result = AgentResult(
                    query      = query,
                    answer     = answer,
                    sources    = sources,
                    trace      = trace,
                    iterations = iteration,
                )
                if self.cache and not no_cache:
                    self.cache.set(f"agent:{query}", 0, result.__dict__)
                return result

            observation = self.executor.execute(tool_call)
            tool_call.result = observation

            # Track sources
            for line in observation.split("\n"):
                if line.startswith("[") and "·" in line:
                    parts = line.strip("[]").split("·")
                    if parts:
                        fname = parts[0].strip()
                        loc   = parts[1].strip() if len(parts) > 1 else ""
                        score = parts[2].strip() if len(parts) > 2 else ""
                        if not any(s["file"] == fname and s.get("loc") == loc for s in sources):
                            sources.append({"file": fname, "loc": loc, "score": score})

            collected.append(f"[Tool: {tool_call.name} | query: {tool_call.args}]\n{observation}")

            step = AgentStep(
                iteration   = iteration,
                thought     = thought,
                tool_call   = tool_call,
                observation = observation[:300],
            )
            trace.append(step)

        # Fallback if loop exhausted without calling answer
        answer = self._extract_best_answer(collected, query)
        return AgentResult(
            query      = query,
            answer     = answer,
            sources    = sources,
            trace      = trace,
            iterations = self.max_iterations,
        )

    def _format_trace(self, trace: List[AgentStep]) -> str:
        if not trace:
            return "None"
        lines = []
        for step in trace[-3:]:  # last 3 steps only
            lines.append(f"Iteration {step.iteration}: {step.thought[:120]}")
            if step.tool_call:
                lines.append(f"  Tool: {step.tool_call.name}({step.tool_call.args})")
                lines.append(f"  Result: {step.observation[:120]}")
        return "\n".join(lines)

    def _extract_best_answer(self, collected: List[str], query: str) -> str:
        if not collected:
            return "I could not find relevant information to answer your question."
        # Return first meaningful chunk as fallback
        for c in collected:
            lines = [l for l in c.split("\n") if not l.startswith("[Tool:")]
            if lines:
                return " ".join(lines)[:600] + "\n\n[Note: Answer derived from partial retrieval]"
        return "Insufficient information found."

    def print_trace(self, result: AgentResult) -> None:
        """Pretty-print the agent's reasoning trace."""
        print(f"\n{'═'*60}")
        print(f"  Query      : {result.query}")
        print(f"  Iterations : {result.iterations}")
        print(f"  Cache hit  : {result.cache_hit}")
        print(f"{'═'*60}")
        for step in result.trace:
            print(f"\n  [{step.iteration}] {step.thought[:100]}")
            if step.tool_call:
                print(f"      → {step.tool_call.name}({json.dumps(step.tool_call.args)[:80]})")
                print(f"      ← {step.observation[:120]}")
        print(f"\n{'─'*60}")
        print(f"  Answer:\n  {result.answer[:500]}")
        print(f"\n  Sources:")
        for s in result.sources:
            print(f"    • {s['file']}  {s.get('loc','')}  {s.get('score','')}")
        print()


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, chromadb, os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from generation.rag import (
        HuggingFaceEmbedder, RedisCache,
        CHROMA_PATH, COLLECTION, PAGE_COLLECTION, HF_MODEL,
        LLM_PROVIDER, OLLAMA_MODEL, GEMINI_MODEL, GOOGLE_API_KEY,
    )

    parser = argparse.ArgumentParser(description="Agentic RAG — LLM-driven iterative retrieval")
    parser.add_argument("query",    nargs="?", default=None, help="Question to answer")
    parser.add_argument("--trace",  action="store_true",    help="Print full reasoning trace")
    parser.add_argument("--iter",   type=int, default=MAX_ITERATIONS, help="Max iterations")
    args = parser.parse_args()

    embedder  = HuggingFaceEmbedder(HF_MODEL)
    db        = chromadb.PersistentClient(path=CHROMA_PATH)
    chunk_col = db.get_or_create_collection(COLLECTION, embedding_function=embedder,
                                             metadata={"hnsw:space": "cosine"})
    try:
        page_col = db.get_collection(PAGE_COLLECTION, embedding_function=embedder)
    except Exception:
        page_col = chunk_col

    try:    cache = RedisCache()
    except: cache = None

    agent = AgenticRAG(
        chunk_col      = chunk_col,
        page_col       = page_col,
        cache          = cache,
        llm_provider   = LLM_PROVIDER,
        ollama_model   = OLLAMA_MODEL,
        gemini_model   = GEMINI_MODEL,
        google_api_key = GOOGLE_API_KEY,
        max_iterations = args.iter,
    )

    if args.query:
        result = agent.run(args.query)
        if args.trace:
            agent.print_trace(result)
        else:
            print(f"\n{result.answer}\n")
        sys.exit(0)

    # Interactive REPL
    print("\n  Agentic RAG REPL  (type 'exit' to quit, add --trace for reasoning)")
    print("  " + "─"*50)
    while True:
        try:
            query = input("\n  🤖 Ask: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!"); break
        if not query: continue
        if query.lower() in {"exit","quit","q"}: print("  Bye!"); break
        result = agent.run(query)
        agent.print_trace(result)
