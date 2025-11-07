"""
LangGraph + Ollama: Agentic RAG Web Search (Single File, Small LLMs)
===================================================================

This script implements an **agentic RAG** loop that plans → searches the web (DuckDuckGo via `ddgs`) → scrapes → chunks → embeds (Ollama) → retrieves → answers with sources, and can **reflect** to do another pass.

Key fixes & choices in this refactor:
- ✅ Uses **`ddgs`** (new name of `duckduckgo_search`).
- ✅ **Reflect node** always returns state; routing to `search` or `END` is done via a conditional edge → avoids `InvalidUpdateError: Expected dict, got __end__`.
- ✅ **Streaming** rehydrates `AgentState` from dict events → avoids `AttributeError: 'dict' object has no attribute ...`.
- ✅ No non-JSON objects stored in state (index is rebuilt inside the answer node to keep things robust across versions).
- ✅ **FAISS optional**: falls back to a pure NumPy cosine similarity index if `faiss-cpu` isn't installed.

Prereqs
-------
1) **Ollama** installed & running locally: https://ollama.com
   - Pull your models, e.g.: `ollama pull llama3.2:3b` and `ollama pull qwen3-embedding:0.6b`
2) Python deps:
   ```bash
   pip install -U langgraph langchain-core ddgs beautifulsoup4 requests numpy pydantic rich
   # Optional (faster retrieval):
   pip install faiss-cpu
   ```

Run
---
```bash
python langgraph_ollama_rag_web_agent.py \
  --model llama3.2:3b \
  --embed-model qwen3-embedding:0.6b \
  --question "What are the key differences between ARM and x86 for laptop efficiency?" \
  --max-iters 2
```
"""

from __future__ import annotations
import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS  # renamed package of duckduckgo_search
from pydantic import BaseModel
from rich import print

# Try FAISS (optional)
try:  # pragma: no cover
    import faiss  # type: ignore
except Exception:  # noqa: BLE001
    faiss = None

# LangGraph core
from langgraph.graph import StateGraph, END

# -----------------------------
# Ollama simple client helpers
# -----------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def ollama_chat(model: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    """Minimal chat wrapper for Ollama /api/chat. Returns assistant text content."""
    url = f"{OLLAMA_HOST}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns last message under key "message"
    return data["message"]["content"]


def ollama_embed(model: str, texts: List[str]) -> np.ndarray:
    """Call Ollama embeddings API for each text and stack into (n, d) array."""
    url = f"{OLLAMA_HOST}/api/embeddings"
    out: List[np.ndarray] = []
    for t in texts:
        r = requests.post(url, json={"model": model, "prompt": t}, timeout=120)
        r.raise_for_status()
        out.append(np.array(r.json()["embedding"], dtype=np.float32))
    return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)


# -----------------------------
# Utilities
# -----------------------------

def clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += step
    return chunks


@dataclass
class DocChunk:
    url: str
    title: str
    chunk: str
    idx: int


# -----------------------------
# Simple vector index (FAISS if available, else NumPy)
# -----------------------------
class VectorIndex:
    def __init__(self, dim: int):
        self.use_faiss = faiss is not None
        self.dim = dim
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(dim)  # cosine via normalized IP
        else:
            self.matrix: Optional[np.ndarray] = None
        self.meta: List[DocChunk] = []

    def add(self, vecs: np.ndarray, metas: List[DocChunk]):
        # normalize to unit vectors
        if vecs.size == 0:
            return
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        vecs = vecs / norms
        if self.use_faiss:
            self.index.add(vecs)
        else:
            self.matrix = vecs if self.matrix is None else np.vstack([self.matrix, vecs])
        self.meta.extend(metas)

    def search(self, q: np.ndarray, k: int = 5) -> List[Tuple[float, DocChunk]]:
        if q.size == 0 or not self.meta:
            return []
        # normalize query
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        if self.use_faiss:
            D, I = self.index.search(q, k)
            out: List[Tuple[float, DocChunk]] = []
            for score, idx in zip(D[0], I[0]):
                if idx == -1:
                    continue
                out.append((float(score), self.meta[int(idx)]))
            return out
        else:
            assert self.matrix is not None
            sims = (self.matrix @ q.T).ravel()
            idxs = np.argsort(-sims)[:k]
            return [(float(sims[i]), self.meta[i]) for i in idxs]


# -----------------------------
# Agent State for LangGraph
# -----------------------------
class AgentState(BaseModel):
    question: str
    plan: List[str] = []                # search queries
    results: List[Dict[str, Any]] = []  # search results metadata
    pages: Dict[str, str] = {}          # url -> cleaned text
    chunks: List[DocChunk] = []
    citations: List[str] = []
    answer: str = ""
    iter: int = 0
    reflect_again: bool = False


# -----------------------------
# Node Implementations
# -----------------------------

def node_plan(state: AgentState, model: str) -> AgentState:
    prompt = (
        "You are a research planner. Given the user's question, craft up to 3 web "
        "search queries that will likely lead to authoritative sources. "
        "Output as a JSON list of strings only.\n\n"
        f"Question: {state.question}\n"
    )
    content = ollama_chat(
        model,
        [
            {"role": "system", "content": "Be concise and return only JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    try:
        queries = json.loads(content)
        if not isinstance(queries, list):
            queries = [state.question]
    except Exception:  # noqa: BLE001
        queries = [state.question]
    state.plan = [q for q in queries if isinstance(q, str) and q.strip()][:3]
    if not state.plan:
        state.plan = [state.question]
    return state


def node_search(state: AgentState) -> AgentState:
    hits: List[Dict[str, Any]] = []
    with DDGS() as ddgs:
        for q in state.plan:
            for r in ddgs.text(q, region="wt-wt", safesearch="moderate", max_results=5):
                hits.append({
                    "title": r.get("title", ""),
                    "href": r.get("href", ""),
                    "body": r.get("body", ""),
                })
    # de-dup by href
    seen: set[str] = set()
    dedup: List[Dict[str, Any]] = []
    for r in hits:
        href = r.get("href")
        if href and href not in seen:
            seen.add(href)
            dedup.append(r)
    state.results = dedup[:12]
    return state


def node_fetch(state: AgentState, timeout: int = 15) -> AgentState:
    pages: Dict[str, str] = {}
    for r in state.results:
        url = r.get("href")
        if not url:
            continue
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0 (RAG-Agent/1.0)"},
            )
            if resp.status_code == 200 and resp.text:
                pages[url] = clean_text(resp.text)[:300_000]  # cap per page
        except Exception:  # noqa: BLE001
            continue
    state.pages = pages
    return state


def node_chunk(state: AgentState) -> AgentState:
    chunks: List[DocChunk] = []
    for url, text in state.pages.items():
        title = next((r["title"] for r in state.results if r.get("href") == url), url)
        for i, ch in enumerate(chunk_text(text)):
            if ch:
                chunks.append(DocChunk(url=url, title=title, chunk=ch, idx=i))
    state.chunks = chunks
    return state


def node_answer(state: AgentState, model: str, embed_model: str, k: int = 6) -> AgentState:
    if not state.chunks:
        state.answer = "I couldn't find enough material to answer."
        return state

    # Build ephemeral index for retrieval (no heavy objects on state)
    texts = [c.chunk for c in state.chunks]
    vecs = ollama_embed(embed_model, texts)
    if vecs.size == 0:
        state.answer = "Embedding model returned no vectors; cannot retrieve."
        return state
    index = VectorIndex(vecs.shape[1])
    index.add(vecs, state.chunks)

    # Query
    q_vec = ollama_embed(embed_model, [state.question])
    results = index.search(q_vec, k=k)

    # Build context + citations
    context_blocks: List[str] = []
    citations: List[str] = []
    for score, doc in results:
        context_blocks.append(
            f"[Source: {doc.title} | {doc.url} | score={score:.3f}]\n{doc.chunk}"
        )
        if doc.url not in citations:
            citations.append(doc.url)
    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a precise researcher. Using only the CONTEXT below, write a clear, concise answer (5-10 sentences). Include inline numeric citations like [1], [2] that map to the Sources list you will print at the end.
If the context is insufficient, say so explicitly.

QUESTION:\n{state.question}

CONTEXT:\n{context}

Return strictly in this format:
ANSWER: <your answer with [n] citations>
SOURCES:\n1. <url>\n2. <url>\n...
"""
    reply = ollama_chat(
        model,
        [
            {"role": "system", "content": "Be accurate. If unsure, say you are unsure."},
            {"role": "user", "content": prompt},
        ],
    )
    state.answer = reply
    state.citations = citations
    return state


def node_reflect(state: AgentState, model: str, max_iters: int) -> AgentState:
    # increment iteration counter
    state.iter += 1

    # If we've hit the cap, no second opinion
    if state.iter >= max_iters:
        state.reflect_again = False
        return state

    prompt = f"""
Given the QUESTION and the DRAFT ANSWER below, determine if another search pass could significantly improve accuracy or add important missing angles. Answer exactly "YES" or "NO".

QUESTION: {state.question}
DRAFT ANSWER:\n{state.answer}
"""
    verdict = ollama_chat(
        model,
        [
            {"role": "system", "content": "Answer strictly YES or NO."},
            {"role": "user", "content": prompt},
        ],
    ).strip().upper()

    state.reflect_again = verdict.startswith("Y")
    return state


# -----------------------------
# Build the LangGraph
# -----------------------------

def build_graph(llm_model: str, embed_model: str, max_iters: int = 2):
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("plan", lambda s: node_plan(s, llm_model))
    graph.add_node("search", node_search)
    graph.add_node("fetch", node_fetch)
    graph.add_node("chunk", node_chunk)
    graph.add_node("answer", lambda s: node_answer(s, llm_model, embed_model))
    graph.add_node("reflect", lambda s: node_reflect(s, llm_model, max_iters))

    # Edges
    graph.set_entry_point("plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "fetch")
    graph.add_edge("fetch", "chunk")
    graph.add_edge("chunk", "answer")
    graph.add_edge("answer", "reflect")

    def route_after_reflect(s: AgentState):
        if s.reflect_again and s.iter < max_iters:
            # Loop back with a broad query to diversify sources
            s.plan = [s.question]
            return "search"
        return END

    graph.add_conditional_edges("reflect", route_after_reflect, {"search": "search", END: END})
    return graph.compile()


# -----------------------------
# CLI / Runner
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="LangGraph + Ollama agentic RAG web search")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama chat model")
    parser.add_argument("--embed-model", default="qwen3-embedding:0.6b", help="Ollama embedding model")
    parser.add_argument("--question", required=True, help="User question to research")
    parser.add_argument("--max-iters", type=int, default=2, help="Max reflect cycles (>=1)")
    args = parser.parse_args()

    # Warmups / checks (fail early if Ollama not reachable)
    try:
        _ = ollama_chat(args.model, [{"role": "user", "content": "ping"}])
    except Exception as e:  # noqa: BLE001
        print("[red]Error contacting Ollama. Is the server running and the model pulled?[/red]")
        raise

    # Build graph and initial state
    app = build_graph(args.model, args.embed_model, max_iters=max(1, args.max_iters))
    state = AgentState(question=args.question)

    # Stream progress and capture final snapshot
    final_snapshot: Optional[AgentState] = None
    for event in app.stream(state, stream_mode="values"):
        snap = AgentState(**event)  # rehydrate from dict
        final_snapshot = snap
        if snap.plan and not snap.results:
            print(f"[bold]Plan:[/bold] {snap.plan}")
        if snap.results and not snap.pages:
            print(f"[bold]Search hits:[/bold] {len(snap.results)}")
        if snap.pages and not snap.chunks:
            print(f"[bold]Fetched pages:[/bold] {len(snap.pages)}")
        if snap.answer and snap.iter == 0:
            print("[bold]Draft answer produced.[/bold]")

    # Print final
    if final_snapshot is None:
        print("[red]No output produced.[/red]")
        return

    print("\n[bold green]ANSWER[/bold green]\n" + final_snapshot.answer)
    if final_snapshot.citations:
        print("\n[bold]Collected Sources:[/bold]")
        for i, url in enumerate(final_snapshot.citations, 1):
            print(f"{i}. {url}")


if __name__ == "__main__":
    main()
