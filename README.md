# LangGraph + Ollama: Agentic RAG for Web Search (Small LLMs)

> **TL;DR**: A single-file, production-ready learning example that uses **LangGraph** to orchestrate an *agentic RAG* loop with **Ollama** small models. It plans queries → searches the web (via **ddgs**) → scrapes → chunks → embeds → retrieves → writes an answer with sources, then optionally **reflects** and loops.

---

## ✨ Features

* **Agentic loop** with a reflect pass (plan → search → fetch → chunk → retrieve → answer → *reflect*).
* **Small-model friendly**: tested with local models like `llama3.2:3b`, `qwen2.5:3b`, `phi3:mini`.
* **Embeddings via Ollama** (default `qwen3-embedding:0.6b`, but any local embed model works).
* **Optional Streamlit UI** for point-and-click research sessions.
* **Web search** using the actively maintained **`ddgs`** package (the successor to `duckduckgo_search`).
* **Vector search** with **FAISS** if available, otherwise a NumPy cosine fallback.
* **Clean streaming** and **safe routing**: no `__end__` write errors; event snapshots rehydrated into Pydantic state.
* Single-file for easy reading, hacking, and demos.

---

## 🧩 Architecture at a Glance

```
flowchart LR
    A[User Question] --> B[Plan]
    B --> C[Search]
    C --> D[Fetch]
    D --> E[Chunk]
    E --> F[Answer (RAG)]
    F --> G{Reflect?}
    G -- Yes --> C
    G -- No --> H[Final Answer + Sources]
```

**LangGraph nodes** (see function names in code):

* `node_plan` → crafts up to 3 search queries (JSON-only output enforced by a small LLM prompt)
* `node_search` → web results via `ddgs`
* `node_fetch` → HTML fetch + **BeautifulSoup** cleaning
* `node_chunk` → simple word-based chunking with overlap
* `node_answer` → build ephemeral index, retrieve top-k, synthesize answer + sources
* `node_reflect` → asks the model whether another pass would improve the answer

**Routing**:

* Conditional edge after `reflect`: if `reflect_again` **and** `iter < max_iters`, jump back to `search`; else end.

---

## 📦 Requirements

* **Python** 3.10+
* **Ollama** running locally: [https://ollama.com](https://ollama.com)
* Optionally **FAISS** (`faiss-cpu`) for faster retrieval.

### Install

Use pip or **uv** (fast Python package/venv manager).

```bash
# Using uv (creates .venv automatically and reads pyproject.toml)
uv venv
source .venv/bin/activate
uv pip install -e .
# (Alternatively: uv pip install -r requirements.txt)

# Using pip directly (mirrors requirements.txt pins)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> ℹ️ Dependencies are pinned so they play nicely with common ecosystem tools
> (e.g., `langchain-qdrant` needs `langchain-core<0.3`, `numba` needs `numpy<2.2`).

### Pull models with Ollama

```bash
# Chat model (pick one you have locally)
ollama pull llama3.2:3b
# or: ollama pull qwen2.5:3b
# or: ollama pull phi3:mini

# Embedding model
ollama pull qwen3-embedding:0.6b
```

---

## 🚀 Quick Start

```bash
python langgraph_ollama_rag_web_agent.py \
  --model llama3.2:3b \
  --embed-model qwen3-embedding:0.6b \
  --question "What are the key differences between ARM and x86 for laptop efficiency?" \
  --max-iters 2
```

**CLI options**

* `--model` → Ollama chat model (default `llama3.2:3b`)
* `--embed-model` → Ollama embedding model (default `qwen3-embedding:0.6b`)
* `--question` → your research question *(required)*
* `--max-iters` → reflect loop budget (default `2`, minimum `1`)

**Environment**

* `OLLAMA_HOST` (optional) → defaults to `http://localhost:11434`

---

## 🖥️ Streamlit UI

Prefer an interactive interface? Launch the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

Use the sidebar to pick chat/embedding models, tune retrieval knobs, and fire questions without touching the CLI. The UI calls the same underlying agent pipeline, so requirements are identical (Ollama running locally plus the pulled models).

---

## 🔍 How It Works (brief)

1. **Plan**: The chat model outputs up to 3 JSON search queries.
2. **Search**: `ddgs` returns top results; we de-duplicate by URL.
3. **Fetch**: pages are downloaded and cleaned (HTML → text) with BeautifulSoup.
4. **Chunk**: simple word-based chunking with overlap for recall.
5. **Embed + Retrieve**: build an *ephemeral* index (FAISS if available, else NumPy cosine) and pull top-k chunks.
6. **Answer**: the chat model writes a concise answer with numeric inline citations `[1]`, `[2]` mapping to source URLs.
7. **Reflect**: the model returns **YES/NO** to decide whether to loop for another pass.

The state model is a Pydantic `AgentState`, and the streaming loop **rehydrates** dict snapshots with `AgentState(**event)`.

---

## 🛠️ Troubleshooting

**`RuntimeWarning: This package (duckduckgo_search) has been renamed to ddgs`**
→ Install `ddgs` and use the new import:

```bash
pip uninstall -y duckduckgo-search
pip install -U ddgs
```

```python
from ddgs import DDGS
```

**`AttributeError: 'dict' object has no attribute 'plan'` while streaming**
→ Stream events are dicts. Rehydrate them:

```python
for event in app.stream(state, stream_mode="values"):
    snap = AgentState(**event)
```

**`InvalidUpdateError: Expected dict, got __end__`**
→ Don’t return `END` from nodes. The reflect node returns **state**, and the **conditional edge** returns `"search"` or `END`.

**`faiss-cpu required`**
→ You can install `faiss-cpu` or switch to the NumPy-based vector index variant (see alternative implementation in README notes or adapt code to a fallback).

**No vectors / Empty embeddings**
→ Ensure `ollama pull qwen3-embedding:0.6b` and that the embedding endpoint is reachable.

---

## ⚙️ Performance & Quality Tips

* **Prefer FAISS** for larger corpora or multi-pass runs.
* **Chunk size & overlap**: tune `chunk_size=800`, `overlap=100` per your domain.
* **k (top-k)** retrieval: start with `k=6` and adjust.
* **Reflection budget** (`--max-iters`) balances quality vs. latency.
* **User-Agent**: set a descriptive UA string when fetching pages; consider timeouts & retries.
* **Caching** (future): add on-disk cache for pages & embeddings to speed up repeats.

---

## 🔒 Ethics & Compliance

* Respect websites’ **robots.txt** and terms of service.
* Rate-limit and avoid scraping sites that forbid automated access.
* Do not collect or store sensitive personal data; follow your org’s data policies.

---

## 🧪 Try Multiple Small Models

Quick shell loop to compare models on the same question:

```bash
for m in "phi3:mini" "qwen2.5:3b" "llama3.2:3b"; do
  echo "\n===== $m ====="
  python langgraph_ollama_rag_web_agent.py \
    --model "$m" \
    --embed-model qwen3-embedding:0.6b \
    --question "What are the key differences between ARM and x86 for laptop efficiency?" \
    --max-iters 2
done
```

Capture outputs to files and review source coverage and answer concision.

---

## 🧱 File Layout

Single file (as requested):

* `langgraph_ollama_rag_web_agent.py` → all logic in one place.

Key sections in the script:

* **Ollama helpers** (`ollama_chat`, `ollama_embed`)
* **Utilities** (`clean_text`, `chunk_text`)
* **Vector index** (FAISS or NumPy)
* **State** (`AgentState`)
* **Nodes** (plan, search, fetch, chunk, answer, reflect)
* **Graph builder** (`build_graph`)
* **Runner** (`main`)

---

## 🗺️ Roadmap Ideas

* Disk cache for **pages** and **embeddings**.
* Domain allow/deny lists; per-source caps.
* Richer HTML-to-text cleanup (readability extraction).
* Query diversification heuristics; self-ask.
* Safety filters / NSFW checks.
* Structured **JSON** output with evidence spans.

---

## 🙏 Acknowledgements

* [LangGraph](https://github.com/langchain-ai/langgraph)
* [Ollama](https://ollama.com)
* [ddgs](https://pypi.org/project/ddgs/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)

---
