"""
Streamlit UI for the LangGraph-based Ollama web agent.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import streamlit as st

from langgraph_ollama_rag_web_agent import AgentState, build_graph


@dataclass
class AgentResult:
    answer: str
    sources: List[str]
    plan: List[str]
    iterations: int


def run_agent(question: str, model: str, embed_model: str, max_iters: int) -> AgentResult:
    app = build_graph(model, embed_model, max_iters=max_iters)
    state = AgentState(question=question.strip())

    final_snapshot: Optional[AgentState] = None
    initial_plan: List[str] = []

    for event in app.stream(state, stream_mode="values"):
        snap = AgentState(**event)
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


st.set_page_config(
    page_title="Agentic Web Researcher",
    page_icon="🕸️",
    layout="wide",
)

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
