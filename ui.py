"""
ui.py
-----
Streamlit UI for the AI Ops Intelligence Platform.
Calls the FastAPI backend at localhost:8000 — does NOT do any
PDF processing or LLM calls directly.

Run with:
    streamlit run ui.py

Make sure the FastAPI server is running first:
    uvicorn app.main:app --reload --port 8000
"""

import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AI Ops Intelligence Platform",
    page_icon="🤖",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🤖 AI Ops Intelligence Platform")
st.caption("Production-grade RAG chatbot with MLOps, drift monitoring, cost tracking, and prompt versioning.")

# ── Sidebar — system status ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Status")

    # Health check
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        st.success(f"API: {health['status'].upper()} v{health['version']}")
    except Exception:
        st.error("API: OFFLINE — start uvicorn first")

    # Cost today
    try:
        cost = requests.get(f"{API_BASE}/cost", timeout=3).json()
        st.metric("Today's spend", f"${cost['today_spend_usd']:.4f}")
    except Exception:
        st.metric("Today's spend", "$0.0000")

    # Prompt versions
    try:
        prompts = requests.get(f"{API_BASE}/prompts", timeout=3).json()
        versions = prompts.get("versions", ["v1"])
    except Exception:
        versions = ["v1"]

    st.divider()
    st.header("🔧 Settings")
    prompt_version = st.selectbox("Prompt version", versions, index=0)
    min_score      = st.slider("Min similarity score", 0.5, 1.0, 0.75, 0.05,
                                help="Chunks below this score are dropped. Higher = stricter grounding.")
    top_k          = st.slider("Top-K chunks", 1, 10, 3,
                                help="How many chunks to retrieve per question.")

    st.divider()
    st.header("📊 Model Metrics")
    if st.button("Refresh metrics"):
        try:
            metrics = requests.get(f"{API_BASE}/metrics", timeout=5).json()
            if "faithfulness" in metrics:
                st.metric("Faithfulness",       f"{metrics['faithfulness']:.2f}")
                st.metric("Hallucination rate", f"{metrics['hallucination_rate']:.2f}")
                st.metric("Answer relevancy",   f"{metrics['answer_relevancy']:.2f}")
                st.caption(f"Prompt: {metrics.get('prompt_version','—')} | n={metrics.get('num_samples','—')}")
            else:
                st.info(metrics.get("message", "Run an eval first."))
        except Exception as e:
            st.error(f"Could not fetch metrics: {e}")

    if st.button("▶ Run RAGAS Eval"):
        with st.spinner("Running evaluation..."):
            try:
                result = requests.post(f"{API_BASE}/eval", timeout=120).json()
                st.success(f"Faithfulness: {result.get('faithfulness', 0):.2f}")
                st.info(f"Hallucination rate: {result.get('hallucination_rate', 0):.2f}")
            except Exception as e:
                st.error(f"Eval failed: {e}")


# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📤 Upload PDF", "💬 Ask Questions"])


# ── Tab 1: Upload ──────────────────────────────────────────────────────────
with tab1:
    st.header("📄 Upload and Process PDF")
    st.info("Upload a PDF — it will be chunked, embedded, and stored for Q&A.")

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        if st.button("🚀 Process PDF"):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    response = requests.post(
                        f"{API_BASE}/upload",
                        files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                        timeout=120,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ {result['message']}")
                        col1, col2 = st.columns(2)
                        col1.metric("Filename",   result["filename"])
                        col2.metric("Chunks created", result["num_chunks"])
                        st.session_state["pdf_ready"] = True
                        st.session_state["pdf_name"]  = result["filename"]
                    else:
                        st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Could not reach API: {e}")

    if st.session_state.get("pdf_ready"):
        st.success(f"✅ Ready to answer questions about: **{st.session_state.get('pdf_name', 'uploaded PDF')}**")


# ── Tab 2: Ask Questions ───────────────────────────────────────────────────
with tab2:
    st.header("💬 Ask a Question")

    if not st.session_state.get("pdf_ready"):
        st.warning("⚠️ Upload and process a PDF first in the Upload tab.")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Confidence",    meta.get("confidence", "—"))
                c2.metric("Grounded",      "✅" if meta.get("is_grounded") else "⚠️ No")
                c3.metric("Fallback used", "Yes" if meta.get("used_fallback") else "No")
                c4.metric("Cost",          f"${meta.get('estimated_cost_usd', 0):.5f}")
                c5.metric("Latency",       f"{meta.get('latency_ms', 0):.0f}ms")

                if meta.get("used_fallback"):
                    st.warning(f"⚠️ Fallback reason: {meta.get('fallback_reason', '—')}")

                with st.expander("📄 Retrieved chunks"):
                    for i, chunk in enumerate(meta.get("retrieved_chunks", [])):
                        st.markdown(f"**Chunk {i+1}** (score: {chunk['score']:.3f})")
                        st.text(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])
                        st.divider()

    # Question input
    question = st.chat_input("Ask a question about the uploaded PDF...")

    if question:
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Call API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {
                        "question":       question,
                        "session_id":     "streamlit-session",
                        "top_k":          top_k,
                        "min_score":      min_score,
                        "prompt_version": prompt_version,
                    }
                    response = requests.post(
                        f"{API_BASE}/ask",
                        json=payload,
                        timeout=60,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        st.markdown(answer)

                        # Production metrics row
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("Confidence",    data["confidence"])
                        c2.metric("Grounded",      "✅" if data["is_grounded"] else "⚠️ No")
                        c3.metric("Fallback used", "Yes" if data["used_fallback"] else "No")
                        c4.metric("Cost",          f"${data['estimated_cost_usd']:.5f}")
                        c5.metric("Latency",       f"{data['latency_ms']:.0f}ms")

                        if data["used_fallback"]:
                            st.warning(f"⚠️ Fallback reason: {data['fallback_reason']}")

                        with st.expander("📄 Retrieved chunks"):
                            for i, chunk in enumerate(data["retrieved_chunks"]):
                                st.markdown(f"**Chunk {i+1}** (score: {chunk['score']:.3f})")
                                st.text(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])
                                st.divider()

                        # Save to history
                        st.session_state["messages"].append({
                            "role":    "assistant",
                            "content": answer,
                            "meta":    data,
                        })

                    elif response.status_code == 404:
                        st.error("No PDF uploaded yet. Go to the Upload tab first.")
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the API. Make sure uvicorn is running on port 8000.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

    # Clear chat button
    if st.session_state.get("messages"):
        if st.button("🗑️ Clear chat"):
            st.session_state["messages"] = []
            st.rerun()
