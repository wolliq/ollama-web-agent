"""
Streamlit UI and LangGraph-based Ollama agent in a single script.

Run options:
    streamlit run streamlit_app.py
    python streamlit_app.py --cli --question "..."
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from ddgs import DDGS
from pydantic import BaseModel
from rich import print

from langgraph.graph import END, StateGraph

CLI_MODE = "--cli" in sys.argv
if CLI_MODE:
    try:
        sys.argv.remove("--cli")
    except ValueError:
        pass

# Try FAISS (optional)
try:  # pragma: no cover
    import faiss  # type: ignore
except Exception:  # noqa: BLE001
    faiss = None

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


# -----------------------------
# Ollama client helpers
# -----------------------------
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


class DocChunk(BaseModel):
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
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        if self.use_faiss:
            D, I = self.index.search(q, k)
            out: List[Tuple[float, DocChunk]] = []
            for score, idx in zip(D[0], I[0]):
                if idx == -1:
                    continue
                out.append((float(score), self.meta[int(idx)]))
            return out
        assert self.matrix is not None
        sims = (self.matrix @ q.T).ravel()
        idxs = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.meta[i]) for i in idxs]


# -----------------------------
# Agent State for LangGraph
# -----------------------------
class AgentState(BaseModel):
    question: str
    plan: List[str] = []
    results: List[Dict[str, Any]] = []
    pages: Dict[str, str] = {}
    chunks: List[DocChunk] = []
    citations: List[str] = []
    answer: str = ""
    iter: int = 0
    reflect_again: bool = False


class AgentResult(BaseModel):
    answer: str
    sources: List[str]
    plan: List[str]
    iterations: int


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
                pages[url] = clean_text(resp.text)[:300_000]
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
    texts = [c.chunk for c in state.chunks]
    vecs = ollama_embed(embed_model, texts)
    if vecs.size == 0:
        state.answer = "Embedding model returned no vectors; cannot retrieve."
        return state
    index = VectorIndex(vecs.shape[1])
    index.add(vecs, state.chunks)
    q_vec = ollama_embed(embed_model, [state.question])
    results = index.search(q_vec, k=k)
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
    state.iter += 1
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
    graph.add_node("plan", lambda s: node_plan(s, llm_model))
    graph.add_node("search", node_search)
    graph.add_node("fetch", node_fetch)
    graph.add_node("chunk", node_chunk)
    graph.add_node("answer", lambda s: node_answer(s, llm_model, embed_model))
    graph.add_node("reflect", lambda s: node_reflect(s, llm_model, max_iters))
    graph.set_entry_point("plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "fetch")
    graph.add_edge("fetch", "chunk")
    graph.add_edge("chunk", "answer")
    graph.add_edge("answer", "reflect")

    def route_after_reflect(s: AgentState):
        if s.reflect_again and s.iter < max_iters:
            s.plan = [s.question]
            return "search"
        return END

    graph.add_conditional_edges("reflect", route_after_reflect, {"search": "search", END: END})
    return graph.compile()


# -----------------------------
# Authentication helpers for Streamlit
# -----------------------------
def authenticate(username: str, password: str) -> bool:
    expected_user = os.getenv("STREAMLIT_USERNAME", "admin")
    expected_pass = os.getenv("STREAMLIT_PASSWORD", "changeme")
    return username == expected_user and password == expected_pass


def render_login() -> None:
    st.title("🔐 Agentic Web Researcher")
    st.caption("Please sign in to run web research jobs.")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            if authenticate(username.strip(), password):
                st.session_state.authenticated = True
                st.success("Login successful. Loading app…")
                raise st.rerun()
            st.error("Invalid credentials. Please try again.")


def run_agent(
    question: str,
    model: str,
    embed_model: str,
    max_iters: int,
    on_event: Optional[Callable[[AgentState], None]] = None,
) -> AgentResult:
    app = build_graph(model, embed_model, max_iters=max_iters)
    state = AgentState(question=question.strip())
    final_snapshot: Optional[AgentState] = None
    initial_plan: List[str] = []
    for event in app.stream(state, stream_mode="values"):
        snap = AgentState(**event)
        if on_event:
            on_event(snap)
        if not initial_plan and snap.plan:
            initial_plan = snap.plan
        final_snapshot = snap
    if final_snapshot is None:
        raise RuntimeError("The agent produced no output.")
    plan = final_snapshot.plan or initial_plan
    answer = final_snapshot.answer or "No answer generated."
    sources = final_snapshot.citations or []
    iterations = max(1, final_snapshot.iter)
    return AgentResult(answer=answer, sources=sources, plan=plan, iterations=iterations)


def run_streamlit_ui() -> None:
    st.set_page_config(
        page_title="Agentic Web Researcher",
        page_icon="🕸️",
        layout="wide",
    )
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        render_login()
        st.stop()
    st.title("🕸️ Agentic Web Researcher")
    st.caption(
        "Plan → search → fetch → chunk → embed → retrieve → answer with optional reflection. "
        "Powered by LangGraph + Ollama."
    )
    with st.sidebar:
        st.header("⚙️ Settings")
        llm_model = st.text_input("Chat model", value="llama3.2:3b")
        embed_model = st.text_input("Embedding model", value="qwen3-embedding:0.6b")
        max_iters = st.slider("Max reflect iterations", min_value=1, max_value=4, value=2)
        if st.button("Log out", use_container_width=True):
            st.session_state.authenticated = False
            raise st.rerun()
    question = st.text_area(
        "Ask anything:",
        value="What are the key differences between ARM and x86 for laptop efficiency?",
        height=120,
    )
    run_clicked = st.button("Run web research", type="primary", use_container_width=True)
    if run_clicked:
        if not question.strip():
            st.warning("Please enter a question to research.")
        else:
            with st.spinner("Running agentic flow…"):
                try:
                    result = run_agent(
                        question=question,
                        model=llm_model.strip() or "llama3.2:3b",
                        embed_model=embed_model.strip() or "qwen3-embedding:0.6b",
                        max_iters=max_iters,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Agent run failed: {exc}")
                else:
                    st.success(f"Completed in {result.iterations} iteration(s).")
                    st.subheader("Plan")
                    if result.plan:
                        for idx, query in enumerate(result.plan, 1):
                            st.write(f"{idx}. {query}")
                    else:
                        st.write("_Plan not available._")
                    st.subheader("Answer")
                    st.markdown(result.answer or "_No answer returned._")
                    st.subheader("Sources")
                    if result.sources:
                        for idx, url in enumerate(result.sources, 1):
                            st.markdown(f"{idx}. [{url}]({url})")
                    else:
                        st.write("_No sources captured._")
    else:
        st.info(
            "Enter a research question, adjust optional settings in the sidebar, "
            "and click **Run web research** to interrogate the web."
        )


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="LangGraph + Ollama agentic web search (CLI mode)")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama chat model")
    parser.add_argument("--embed-model", default="qwen3-embedding:0.6b", help="Ollama embedding model")
    parser.add_argument("--question", required=True, help="User question to research")
    parser.add_argument("--max-iters", type=int, default=2, help="Max reflect cycles (>=1)")
    args = parser.parse_args()

    def log_event(snap: AgentState) -> None:
        if snap.plan and not snap.results:
            print(f"[bold]Plan:[/bold] {snap.plan}")
        elif snap.results and not snap.pages:
            print(f"[bold]Search hits:[/bold] {len(snap.results)}")
        elif snap.pages and not snap.chunks:
            print(f"[bold]Fetched pages:[/bold] {len(snap.pages)}")
        elif snap.answer and snap.iter == 0:
            print("[bold]Draft answer produced.[/bold]")

    result = run_agent(
        question=args.question,
        model=args.model,
        embed_model=args.embed_model,
        max_iters=max(1, args.max_iters),
        on_event=log_event,
    )

    print(f"\n[bold green]ANSWER (iterations: {result.iterations})[/bold green]\n{result.answer}")
    if result.plan:
        print("\n[bold]Plan[/bold]")
        for i, step in enumerate(result.plan, 1):
            print(f"{i}. {step}")
    if result.sources:
        print("\n[bold]Collected Sources:[/bold]")
        for i, url in enumerate(result.sources, 1):
            print(f"{i}. {url}")


if CLI_MODE:
    if __name__ == "__main__":
        run_cli()
else:
    run_streamlit_ui()
