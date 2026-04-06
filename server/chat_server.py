"""
Chat server — FastAPI + WebSocket + RAG pipeline.

Serves a chat UI at http://localhost:8000
WebSocket at ws://localhost:8000/ws

Start:
    pip install fastapi uvicorn
    python chat_server.py
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google")

import os, json, asyncio, chromadb, uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ── Import RAG core ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from rag import (
    HuggingFaceEmbedder, RedisCache, retrieve, generate, rag_query,
    CHROMA_PATH, COLLECTION, PAGE_COLLECTION, HF_MODEL, DEFAULT_TOP_K,
    get_context_engine,
)

# ── Init shared resources ──────────────────────────────────────────────────────
embedder  = HuggingFaceEmbedder(HF_MODEL)
db_client = chromadb.PersistentClient(path=CHROMA_PATH)
chunk_col = db_client.get_or_create_collection(COLLECTION,      embedding_function=embedder, metadata={"hnsw:space": "cosine"})
try:
    page_col = db_client.get_collection(PAGE_COLLECTION, embedding_function=embedder)
except Exception:
    print(f"  ⚠ Page index '{PAGE_COLLECTION}' not found — using chunk index only", file=__import__('sys').stderr)
    page_col = chunk_col   # fallback: use chunk collection for both

try:
    cache = RedisCache()
except Exception:
    cache = None

app = FastAPI(title="RAG Chat")

# ── Chat HTML page ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Chat</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #22263a;
    --border: #2e3350; --accent: #6c8fff; --accent2: #a78bfa;
    --text: #e2e8f0; --muted: #7a85a0; --user-bg: #1e3a5f;
    --bot-bg: #1a1d27; --success: #34d399; --warn: #fbbf24;
  }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; height: 100vh; display: flex; flex-direction: column; }

  /* Header */
  header { padding: 14px 24px; background: var(--surface); border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }
  .logo { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 18px; }
  .logo-icon { width: 32px; height: 32px; background: linear-gradient(135deg, var(--accent), var(--accent2)); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 16px; }
  .badges { display: flex; gap: 8px; }
  .badge { padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; border: 1px solid; }
  .badge-green { color: var(--success); border-color: var(--success); background: rgba(52,211,153,.1); }
  .badge-blue  { color: var(--accent);  border-color: var(--accent);  background: rgba(108,143,255,.1); }
  .badge-muted { color: var(--muted);   border-color: var(--border);  background: transparent; }
  #status-badge { transition: all .3s; }

  /* Layout */
  .container { display: flex; flex: 1; overflow: hidden; }

  /* Sidebar */
  aside { width: 260px; background: var(--surface); border-right: 1px solid var(--border); padding: 20px 16px; display: flex; flex-direction: column; gap: 20px; overflow-y: auto; flex-shrink: 0; }
  .sidebar-section h3 { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 10px; }
  .control-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
  .control-row label { font-size: 13px; color: var(--text); }
  .control-row span  { font-size: 12px; color: var(--accent); font-weight: 600; }
  input[type=range] { width: 100%; accent-color: var(--accent); margin: 6px 0 2px; }
  select, .select { width: 100%; background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 7px 10px; border-radius: 6px; font-size: 13px; margin-top: 4px; outline: none; }
  select:focus { border-color: var(--accent); }
  .toggle-row { display: flex; align-items: center; justify-content: space-between; padding: 6px 0; }
  .toggle-label { font-size: 13px; color: var(--text); }
  .toggle { position: relative; width: 36px; height: 20px; }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .slider { position: absolute; inset: 0; background: var(--border); border-radius: 20px; cursor: pointer; transition: .3s; }
  .slider::before { content: ""; position: absolute; width: 14px; height: 14px; left: 3px; top: 3px; background: white; border-radius: 50%; transition: .3s; }
  input:checked + .slider { background: var(--accent); }
  input:checked + .slider::before { transform: translateX(16px); }
  .btn-flush { width: 100%; padding: 8px; background: rgba(239,68,68,.15); border: 1px solid rgba(239,68,68,.3); color: #ef4444; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all .2s; }
  .btn-flush:hover { background: rgba(239,68,68,.25); }

  /* Chat */
  main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #messages { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 18px; scroll-behavior: smooth; }
  #messages::-webkit-scrollbar { width: 6px; }
  #messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  .msg { display: flex; gap: 12px; max-width: 820px; animation: fadeIn .25s ease; }
  .msg.user  { flex-direction: row-reverse; margin-left: auto; }
  @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:none; } }

  .avatar { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 16px; flex-shrink: 0; }
  .avatar.user { background: var(--user-bg); }
  .avatar.bot  { background: linear-gradient(135deg, var(--accent), var(--accent2)); }

  .bubble { padding: 12px 16px; border-radius: 12px; line-height: 1.6; font-size: 14px; max-width: 680px; }
  .msg.user  .bubble { background: var(--user-bg); border-bottom-right-radius: 2px; }
  .msg.bot   .bubble { background: var(--bot-bg); border: 1px solid var(--border); border-bottom-left-radius: 2px; }
  .bubble p  { margin-bottom: 8px; }
  .bubble p:last-child { margin: 0; }

  .cache-tag { display: inline-flex; align-items: center; gap: 4px; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 10px; margin-bottom: 6px; }
  .cache-hit  { background: rgba(52,211,153,.15); color: var(--success); border: 1px solid rgba(52,211,153,.3); }
  .cache-miss { background: rgba(108,143,255,.15); color: var(--accent);  border: 1px solid rgba(108,143,255,.3); }

  .sources { margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 4px; }
  .source-item { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--muted); }
  .source-item .src-file { color: var(--accent2); font-weight: 600; }
  .score-bar { width: 40px; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
  .gw-list { margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(239,68,68,.2); }
  .gw-item { font-size: 11px; color: #f87171; background: rgba(239,68,68,.07); padding: 3px 8px; border-radius: 4px; margin-bottom: 3px; }
  .ctx-steps { margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 3px; }
  .ctx-row { display: flex; gap: 8px; font-size: 11px; }
  .ctx-label { color: var(--muted); min-width: 90px; }
  .ctx-val { color: var(--accent2); }
  .score-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); border-radius: 2px; }

  .typing { display: flex; gap: 4px; align-items: center; padding: 4px 0; }
  .typing span { width: 7px; height: 7px; background: var(--muted); border-radius: 50%; animation: bounce .9s infinite; }
  .typing span:nth-child(2) { animation-delay: .15s; }
  .typing span:nth-child(3) { animation-delay: .30s; }
  @keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-6px)} }

  /* Input bar */
  .input-bar { padding: 16px 24px; background: var(--surface); border-top: 1px solid var(--border); display: flex; gap: 10px; align-items: flex-end; flex-shrink: 0; }
  #input { flex: 1; background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 10px 14px; border-radius: 10px; font-size: 14px; resize: none; min-height: 42px; max-height: 140px; outline: none; line-height: 1.5; transition: border-color .2s; font-family: inherit; }
  #input:focus { border-color: var(--accent); }
  #input::placeholder { color: var(--muted); }
  #send { width: 42px; height: 42px; background: var(--accent); border: none; border-radius: 10px; color: white; font-size: 18px; cursor: pointer; transition: all .2s; flex-shrink: 0; display: flex; align-items: center; justify-content: center; }
  #send:hover { background: var(--accent2); transform: scale(1.05); }
  #send:disabled { opacity: .4; cursor: not-allowed; transform: none; }
  .hint { font-size: 11px; color: var(--muted); padding: 6px 24px 2px; text-align: right; }

  /* Welcome */
  .welcome { margin: auto; text-align: center; max-width: 420px; opacity: .7; }
  .welcome h2 { font-size: 22px; margin-bottom: 8px; }
  .welcome p  { font-size: 14px; color: var(--muted); line-height: 1.6; }
  .suggestions { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 16px; }
  .suggestion { padding: 7px 14px; background: var(--surface2); border: 1px solid var(--border); border-radius: 20px; font-size: 12px; color: var(--muted); cursor: pointer; transition: all .2s; }
  .suggestion:hover { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>
<header>
  <div class="logo">
    <div class="logo-icon">🧠</div>
    RAG Chat
  </div>
  <div class="badges">
    <span class="badge badge-blue" id="model-badge">Loading...</span>
    <span class="badge badge-green" id="status-badge">⚡ Connected</span>
    <span class="badge badge-muted" id="cache-badge">Cache: —</span>
  </div>
</header>

<div class="container">
  <aside>
    <div class="sidebar-section">
      <h3>Retrieval</h3>
      <div class="control-row">
        <label>Top-K chunks</label>
        <span id="topk-val">3</span>
      </div>
      <input type="range" id="topk" min="1" max="10" value="3">
      <div style="margin-top:10px">
        <div class="control-row"><label>Search mode</label></div>
        <select id="mode">
          <option value="both" selected>Both (page + chunk)</option>
          <option value="chunk">Chunk only</option>
          <option value="page">Page only</option>
        </select>
      </div>
    </div>

    <div class="sidebar-section">
      <h3>Options</h3>
      <div class="toggle-row">
        <span class="toggle-label">Skip cache</span>
        <label class="toggle">
          <input type="checkbox" id="no-cache">
          <span class="slider"></span>
        </label>
      </div>
      <div class="toggle-row">
        <span class="toggle-label">Show sources</span>
        <label class="toggle">
          <input type="checkbox" id="show-sources" checked>
          <span class="slider"></span>
        </label>
      </div>
    </div>


    <div class="sidebar-section">
      <h3>Cache</h3>
      <button class="btn-flush" onclick="flushCache()">🗑 Flush Cache</button>
      <button class="btn-flush" style="margin-top:6px;border-color:rgba(167,139,250,.3);color:#a78bfa;background:rgba(167,139,250,.1)" onclick="clearMemory()">🧹 Clear Memory</button>
      <div style="margin-top:8px;font-size:12px;color:var(--muted)" id="cache-info">—</div>
    </div>
  </aside>

  <main>
    <div id="messages">
      <div class="welcome" id="welcome">
        <h2>👋 Ask your documents</h2>
        <p>This RAG chat searches your indexed documents and generates answers using the configured LLM.</p>
        <div class="suggestions">
          <div class="suggestion" onclick="ask(this)">What is RAG?</div>
          <div class="suggestion" onclick="ask(this)">Vector database options</div>
          <div class="suggestion" onclick="ask(this)">Python programming tips</div>
          <div class="suggestion" onclick="ask(this)">How does semantic search work?</div>
        </div>
      </div>
    </div>
    <div class="hint">Press Enter to send · Shift+Enter for new line</div>
    <div class="input-bar">
      <textarea id="input" placeholder="Ask anything about your documents..." rows="1"></textarea>
      <button id="send" onclick="sendMessage()">↑</button>
    </div>
  </main>
</div>

<script>
  let ws, waiting = false;

  function connect() {
    ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onopen    = () => { setStatus("Connected", "green"); requestMeta(); };
    ws.onclose   = () => { setStatus("Disconnected", "red"); setTimeout(connect, 2000); };
    ws.onerror   = () => setStatus("Error", "red");
    ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
  }

  function setStatus(text, color) {
    const el = document.getElementById("status-badge");
    el.textContent = (color === "green" ? "⚡ " : "✗ ") + text;
    el.className = "badge " + (color === "green" ? "badge-green" : "badge-muted");
  }

  function requestMeta() {
    ws.send(JSON.stringify({ type: "meta" }));
  }

  function handleMessage(data) {
    if (data.type === "meta") {
      document.getElementById("model-badge").textContent = data.model;
      document.getElementById("cache-badge").textContent =
        data.cache_available ? `Cache: ${data.cached} queries` : "Cache: off";
      document.getElementById("cache-info").textContent =
        data.cache_available ? `TTL: ${data.ttl}s · ${data.cached} cached` : "Redis not available";
      return;
    }
    if (data.type === "answer") {
      removeTyping();
      appendBot(data);
      waiting = false;
      document.getElementById("send").disabled = false;
    }
    if (data.type === "error") {
      removeTyping();
      appendBot({ answer: "⚠ " + data.message, sources: [], cache_hit: false });
      waiting = false;
      document.getElementById("send").disabled = false;
    }
  }


  function ask(el) { document.getElementById("input").value = el.textContent; sendMessage(); }

  function sendMessage() {
    const input = document.getElementById("input");
    const text  = input.value.trim();
    if (!text || waiting) return;

    document.getElementById("welcome")?.remove();
    appendUser(text);
    appendTyping();

    waiting = true;
    document.getElementById("send").disabled = true;
    input.value = "";
    autoResize(input);

    ws.send(JSON.stringify({
      type     : "query",
      query    : text,
      top_k    : parseInt(document.getElementById("topk").value),
      no_cache : document.getElementById("no-cache").checked,
      mode     : document.getElementById("mode").value,
    }));
  }

  function appendUser(text) {
    const msgs = document.getElementById("messages");
    msgs.insertAdjacentHTML("beforeend", `
      <div class="msg user">
        <div class="avatar user">🧑</div>
        <div class="bubble">${escHtml(text)}</div>
      </div>`);
    msgs.scrollTop = msgs.scrollHeight;
  }

  function appendBot(data) {
    const showSrc = document.getElementById("show-sources").checked;
    const cacheTag = data.cache_hit
      ? `<div class="cache-tag cache-hit">⚡ Cache hit</div>`
      : `<div class="cache-tag cache-miss">🔍 Generated</div>`;
    const srcHtml = (showSrc && data.sources?.length)
      ? `<div class="sources">` + data.sources.map(s => `
          <div class="source-item">
            <span class="src-file">${escHtml(s.file)}</span>
            <span>${escHtml(s.location)}</span>
            <div class="score-bar"><div class="score-fill" style="width:${Math.round(s.score*100)}%"></div></div>
            <span>${(s.score*100).toFixed(0)}%</span>
          </div>`).join("") + `</div>`
      : "";

    // Guardrail warnings
    const gWarnings = data.guardrail_warnings || [];
    const gwHtml = gWarnings.length ? `
      <div class="gw-list">
        ${gWarnings.map(w => `<div class="gw-item">⚠ ${escHtml(w)}</div>`).join("")}
      </div>` : "";

    // Context engineering steps
    const steps = data.context_steps || {};
    const ctxHtml = Object.keys(steps).length ? `
      <div class="ctx-steps">
        ${steps.rewritten_query && steps.rewritten_query !== data.query ? `<div class="ctx-row"><span class="ctx-label">↪ Rewritten</span><span class="ctx-val">${escHtml(steps.rewritten_query)}</span></div>` : ""}
        ${steps.after_mmr != null ? `<div class="ctx-row"><span class="ctx-label">MMR chunks</span><span class="ctx-val">${steps.after_mmr}</span></div>` : ""}
        ${steps.after_budget != null ? `<div class="ctx-row"><span class="ctx-label">After budget</span><span class="ctx-val">${steps.after_budget} chunks</span></div>` : ""}
        ${steps.token_estimate != null ? `<div class="ctx-row"><span class="ctx-label">~Tokens</span><span class="ctx-val">${steps.token_estimate}</span></div>` : ""}
      </div>` : "";
    const msgs = document.getElementById("messages");
    msgs.insertAdjacentHTML("beforeend", `
      <div class="msg bot">
        <div class="avatar bot">🧠</div>
        <div class="bubble">
          ${cacheTag}
          <p>${escHtml(data.answer).replace(/\\n/g,"<br>")}</p>
          ${srcHtml}
          ${gwHtml}
          ${ctxHtml}
        </div>
      </div>`);
    msgs.scrollTop = msgs.scrollHeight;
    requestMeta(); // refresh cache count
  }

  function appendTyping() {
    document.getElementById("messages").insertAdjacentHTML("beforeend", `
      <div class="msg bot" id="typing-indicator">
        <div class="avatar bot"></div>
        <div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>
      </div>`);
    document.getElementById("messages").scrollTop = 999999;
  }

  function removeTyping() {
    document.getElementById("typing-indicator")?.remove();
  }

  function clearMemory() {
    ws.send(JSON.stringify({ type: 'clear_memory' }));
    document.getElementById('messages').innerHTML = '';
    document.getElementById('messages').insertAdjacentHTML('beforeend', `<div class='welcome' id='welcome'><h2>🧹 Memory cleared</h2><p>Conversation history has been reset.</p></div>`);
  }

  function flushCache() {
    ws.send(JSON.stringify({ type: "flush_cache" }));
    setTimeout(requestMeta, 300);
  }

  function escHtml(s) {
    return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  }

  function autoResize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  }

  document.getElementById("input").addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  document.getElementById("input").addEventListener("input", e => autoResize(e.target));
  document.getElementById("topk").addEventListener("input", e => {
    document.getElementById("topk-val").textContent = e.target.value;
  });

  connect();
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def chat_page():
    return HTML

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw  = await websocket.receive_text()
            data = json.loads(raw)

            # ── Meta request ──────────────────────────────────────────────────
            if data["type"] == "meta":
                from rag import LLM_PROVIDER, GEMINI_MODEL, OLLAMA_MODEL, CACHE_TTL
                model_label = OLLAMA_MODEL if LLM_PROVIDER == "ollama" else GEMINI_MODEL

                cache_stats = cache.stats() if cache else {}
                await websocket.send_text(json.dumps({
                    "type"           : "meta",
                    "model"          : f"{LLM_PROVIDER} / {model_label}",
                    "cache_available": cache is not None,
                    "cached"         : cache_stats.get("cached_queries", 0),
                    "ttl"            : CACHE_TTL,
                }))

            # ── Query ─────────────────────────────────────────────────────────
            elif data["type"] == "query":
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: rag_query(
                            query     = data["query"],
                            chunk_col = chunk_col,
                            page_col  = page_col,
                            cache     = cache,
                            top_k     = data.get("top_k", DEFAULT_TOP_K),
                            no_cache  = data.get("no_cache", False),
                        )
                    )
                    await websocket.send_text(json.dumps({
                        "type"          : "answer",
                        "answer"        : result["answer"],
                        "cache_hit"     : result["cache_hit"],
                        "sources"       : result["sources"],
                        "context_steps"      : result.get("context_steps", {}),
                        "guardrail_warnings" : result.get("guardrail_warnings", []),
                        "query"         : data["query"],
                    }))
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

            # ── Flush cache ───────────────────────────────────────────────────
            elif data["type"] == "flush_cache":
                n = cache.flush() if cache else 0
                await websocket.send_text(json.dumps({"type": "meta", "flushed": n}))
                # re-send full meta
                continue

    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run("chat_server:app", host="0.0.0.0", port=8000, reload=False)
