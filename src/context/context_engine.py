import re, math
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# ── Token estimation (no tiktoken dependency) ──────────────────────────────────
def estimate_tokens(text: str) -> int:
    """~1 token per 4 chars — fast approximation."""
    return max(1, len(text) // 4)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Conversation Memory
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Turn:
    role   : str   # "user" | "assistant"
    content: str

class ConversationMemory:
    """
    Rolling conversation window with summary compression.
    Keeps the last `max_turns` turns in full; older turns are summarised.
    """
    def __init__(self, max_turns: int = 6, max_tokens: int = 800):
        self.turns     : List[Turn] = []
        self.summary   : str        = ""
        self.max_turns = max_turns
        self.max_tokens = max_tokens

    def add(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))
        if len(self.turns) > self.max_turns:
            self._compress()

    def _compress(self) -> None:
        """Move oldest turn into the running summary."""
        oldest = self.turns.pop(0)
        prefix = f"[{oldest.role.upper()}]: {oldest.content[:200]}"
        self.summary = (self.summary + "\n" + prefix).strip()[-600:]  # cap summary size

    def get_history(self) -> List[Dict]:
        """Return as list of {role, content} dicts for LLM messages."""
        msgs = []
        if self.summary:
            msgs.append({"role": "user",      "content": f"[Conversation so far]\n{self.summary}"})
            msgs.append({"role": "assistant", "content": "Understood. I'll keep that context in mind."})
        for t in self.turns:
            msgs.append({"role": t.role, "content": t.content})
        return msgs

    def get_recent_text(self, n: int = 3) -> str:
        """Return last n turns as plain text for query rewriting."""
        recent = self.turns[-n:]
        return "\n".join(f"{t.role}: {t.content}" for t in recent)

    def clear(self) -> None:
        self.turns   = []
        self.summary = ""

# ─────────────────────────────────────────────────────────────────────────────
# 2. Query Rewriting
# ─────────────────────────────────────────────────────────────────────────────
class QueryRewriter:
    """
    Rewrites/expands the user query using conversation history.
    Uses simple heuristics — no LLM call needed for most cases.
    Falls back to a lightweight LLM rewrite when the query is ambiguous.
    """
    PRONOUNS = {"it", "this", "that", "they", "them", "he", "she", "its", "their"}

    def rewrite(self, query: str, memory: ConversationMemory) -> str:
        if not memory.turns:
            return query  # no history → nothing to resolve

        query_lower = query.lower().split()

        # Heuristic: query contains pronoun or is very short → resolve via history
        needs_rewrite = (
            any(w in self.PRONOUNS for w in query_lower)
            or len(query_lower) <= 4
            or query.endswith("?") and len(query_lower) <= 6
        )

        if not needs_rewrite:
            return query

        # Inject recent context as prefix to make it self-contained
        recent = memory.get_recent_text(n=2)
        last_topic = self._extract_topic(recent)

        if last_topic and any(p in query_lower for p in self.PRONOUNS):
            rewritten = re.sub(
                r'\b(it|this|that|they|them|its|their)\b',
                last_topic,
                query,
                flags=re.IGNORECASE,
                count=1,
            )
            return rewritten if rewritten != query else f"{last_topic} — {query}"

        return query

    def _extract_topic(self, text: str) -> str:
        """Extract last noun phrase as topic (simple heuristic)."""
        words = text.split()
        # grab last 1-3 capitalized or content words
        candidates = [w.strip("?.,!") for w in words[-20:] if len(w) > 3 and w[0].isupper()]
        return candidates[-1] if candidates else ""

# ─────────────────────────────────────────────────────────────────────────────
# 3. Context Compression
# ─────────────────────────────────────────────────────────────────────────────
class ContextCompressor:
    """
    Removes sentences from retrieved chunks that are unlikely to help answer
    the query, reducing noise and token usage.
    """
    def compress(self, chunks: List[dict], query: str, keep_ratio: float = 0.7) -> List[dict]:
        query_words = set(query.lower().split())
        compressed  = []
        for chunk in chunks:
            sentences  = self._split_sentences(chunk["text"])
            scored     = [(s, self._relevance(s, query_words)) for s in sentences]
            # keep top sentences by relevance, always keep at least 1
            threshold  = sorted([sc for _, sc in scored], reverse=True)[
                max(0, int(len(scored) * (1 - keep_ratio)) - 1)
            ] if scored else 0
            kept = [s for s, sc in scored if sc >= threshold]
            compressed.append({**chunk, "text": " ".join(kept) or chunk["text"]})
        return compressed

    def _split_sentences(self, text: str) -> List[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def _relevance(self, sentence: str, query_words: set) -> float:
        words      = set(sentence.lower().split())
        overlap    = len(words & query_words)
        length_pen = math.log(max(len(words), 1) + 1)
        return overlap / length_pen

# ─────────────────────────────────────────────────────────────────────────────
# 4. MMR Re-ranking (Maximal Marginal Relevance)
# ─────────────────────────────────────────────────────────────────────────────
class MMRRanker:
    """
    Maximal Marginal Relevance — balances relevance vs diversity.
    Penalises chunks that are too similar to already-selected chunks.
    lambda_=1.0 → pure relevance, lambda_=0.0 → pure diversity.
    """
    def rank(self, chunks: List[dict], top_k: int, lambda_: float = 0.7) -> List[dict]:
        if len(chunks) <= top_k:
            return chunks

        selected  : List[dict] = []
        remaining : List[dict] = list(chunks)

        while len(selected) < top_k and remaining:
            if not selected:
                # First pick: highest relevance score
                best = max(remaining, key=lambda c: c["score"])
            else:
                best = max(
                    remaining,
                    key=lambda c: (
                        lambda_ * c["score"]
                        - (1 - lambda_) * max(
                            self._text_similarity(c["text"], s["text"])
                            for s in selected
                        )
                    ),
                )
            selected.append(best)
            remaining.remove(best)

        return selected

    def _text_similarity(self, a: str, b: str) -> float:
        """Jaccard similarity on word sets."""
        wa, wb = set(a.lower().split()), set(b.lower().split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Token Budget Manager
# ─────────────────────────────────────────────────────────────────────────────
class TokenBudgetManager:
    """
    Fits retrieved chunks within a token budget.
    Truncates lower-ranked chunks first; never truncates below min_chunk_tokens.
    """
    def __init__(
        self,
        max_context_tokens : int = 3000,
        system_tokens      : int = 200,
        history_tokens     : int = 400,
        answer_tokens      : int = 1024,
        min_chunk_tokens   : int = 50,
    ):
        overhead              = system_tokens + history_tokens + answer_tokens
        self.budget           = max(500, max_context_tokens - overhead)
        self.min_chunk_tokens = min_chunk_tokens

    def fit(self, chunks: List[dict]) -> List[dict]:
        remaining = self.budget
        fitted    = []
        for chunk in chunks:
            tokens = estimate_tokens(chunk["text"])
            if tokens <= remaining:
                fitted.append(chunk)
                remaining -= tokens
            elif remaining >= self.min_chunk_tokens:
                # Truncate this chunk to fit
                ratio    = remaining / tokens
                words    = chunk["text"].split()
                cut      = max(self.min_chunk_tokens, int(len(words) * ratio))
                truncated = " ".join(words[:cut]) + " [...]"
                fitted.append({**chunk, "text": truncated})
                break
            else:
                break
        return fitted

# ─────────────────────────────────────────────────────────────────────────────
# 6. System Prompt Builder
# ─────────────────────────────────────────────────────────────────────────────
class SystemPromptBuilder:
    """
    Assembles a structured system prompt from:
      - Base persona & instructions
      - Retrieval date / session metadata
      - Retrieved context chunks
      - Conversation history
      - Query
    """
    BASE_SYSTEM = (
        "You are a knowledgeable assistant with access to a curated document knowledge base.\n"
        "Rules:\n"
        "  1. Answer ONLY from the provided context. Do not hallucinate.\n"
        "  2. If the answer is not in the context, say exactly: "
        "\"I don't have enough information to answer that.\"\n"
        "  3. Cite your sources by referencing the [Source] tags.\n"
        "  4. Be concise but complete. Use bullet points for lists.\n"
        "  5. If the question is a follow-up, use the conversation history.\n"
    )

    def build_system(self, extra_instructions: str = "") -> str:
        prompt = self.BASE_SYSTEM
        if extra_instructions:
            prompt += f"\nAdditional instructions:\n{extra_instructions}\n"
        return prompt

    def build_user_message(
        self,
        query  : str,
        chunks : List[dict],
        memory : Optional[ConversationMemory] = None,
    ) -> str:
        parts = []

        # Context block
        if chunks:
            ctx_lines = []
            for i, c in enumerate(chunks, 1):
                ctx_lines.append(
                    f"[Source {i}: {c['filename']} · {c['location']} · score {c['score']:.2f}]\n{c['text']}"
                )
            parts.append("=== CONTEXT ===\n" + "\n\n".join(ctx_lines))
        else:
            parts.append("=== CONTEXT ===\n(No relevant documents found)")

        # Conversation history summary
        if memory and memory.summary:
            parts.append(f"=== CONVERSATION SUMMARY ===\n{memory.summary}")

        # Question
        parts.append(f"=== QUESTION ===\n{query}")

        return "\n\n".join(parts)

    def build_messages(
        self,
        query  : str,
        chunks : List[dict],
        memory : Optional[ConversationMemory] = None,
        extra  : str = "",
    ) -> List[Dict]:
        """Return full messages list for multi-turn LLMs (Ollama, OpenAI-style)."""
        messages = [{"role": "system", "content": self.build_system(extra)}]
        if memory:
            messages.extend(memory.get_history())
        messages.append({"role": "user", "content": self.build_user_message(query, chunks, memory)})
        return messages

# ─────────────────────────────────────────────────────────────────────────────
# 7. Context Engine — orchestrates all of the above
# ─────────────────────────────────────────────────────────────────────────────
class ContextEngine:
    """
    Drop-in replacement for the raw retrieve→generate pipeline.
    Adds: query rewriting, MMR ranking, compression, token budgeting,
    conversation memory, and structured prompt assembly.
    """
    def __init__(
        self,
        max_context_tokens : int   = 3000,
        mmr_lambda         : float = 0.7,
        compress_ratio     : float = 0.7,
        max_history_turns  : int   = 6,
    ):
        self.rewriter   = QueryRewriter()
        self.compressor = ContextCompressor()
        self.ranker     = MMRRanker()
        self.budgeter   = TokenBudgetManager(max_context_tokens=max_context_tokens)
        self.prompt_builder = SystemPromptBuilder()
        self.memory     = ConversationMemory(max_turns=max_history_turns)
        self.mmr_lambda = mmr_lambda
        self.compress_ratio = compress_ratio

    def prepare(
        self,
        query       : str,
        raw_chunks  : List[dict],
        top_k       : int = 6,
        extra_instructions: str = "",
    ) -> Dict:
        """
        Full context engineering pipeline.
        Returns dict with:
          - rewritten_query
          - chunks (engineered)
          - messages (ready to send to LLM)
          - token_estimate
          - steps (audit trail)
        """
        steps = {}

        # Step 1: Query rewriting
        rewritten = self.rewriter.rewrite(query, self.memory)
        steps["rewritten_query"] = rewritten

        # Step 2: MMR re-ranking for diversity
        ranked = self.ranker.rank(raw_chunks, top_k=top_k * 2, lambda_=self.mmr_lambda)
        steps["after_mmr"] = len(ranked)

        # Step 3: Context compression
        compressed = self.compressor.compress(ranked, rewritten, keep_ratio=self.compress_ratio)
        steps["after_compression"] = sum(len(c["text"].split()) for c in compressed)

        # Step 4: Token budget fitting
        fitted = self.budgeter.fit(compressed)
        steps["after_budget"] = len(fitted)
        steps["token_estimate"] = sum(estimate_tokens(c["text"]) for c in fitted)

        # Step 5: Build structured messages
        messages = self.prompt_builder.build_messages(
            query  = rewritten,
            chunks = fitted,
            memory = self.memory,
            extra  = extra_instructions,
        )

        return {
            "original_query" : query,
            "rewritten_query": rewritten,
            "chunks"         : fitted,
            "messages"       : messages,
            "token_estimate" : steps["token_estimate"],
            "steps"          : steps,
        }

    def record_answer(self, query: str, answer: str) -> None:
        """Store turn in conversation memory after generation."""
        self.memory.add("user",      query)
        self.memory.add("assistant", answer)

    def clear_memory(self) -> None:
        self.memory.clear()
