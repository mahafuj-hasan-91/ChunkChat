"""
app.py
──────
Main Streamlit application — the entry point for the RAG Chatbot.

This file ties together all the RAG pipeline components:
  PDF Loader → Chunker → Embedder → Vector Store → LLM → UI

Run with: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

# ─── RAG Pipeline Imports ─────────────────────────────────────────────────
from rag.pdf_loader import load_pdf, get_pdf_metadata
from rag.chunker import chunk_text, get_chunk_stats
from rag.embedder import embed_texts, embed_query
from rag.vector_store import (
    get_chroma_client,
    get_or_create_collection,
    store_embeddings,
    query_similar_chunks,
    clear_collection,
    get_collection_info
)
from rag.llm import generate_answer, test_groq_connection

# ─── Load environment variables from .env ────────────────────────────────
load_dotenv()

# ─── App Configuration ────────────────────────────────────────────────────
DATA_DIR = "./data"          # where uploaded PDFs are saved
os.makedirs(DATA_DIR, exist_ok=True)

TOP_K_CHUNKS = 3             # number of similar chunks to retrieve
MAX_HISTORY = 5              # number of Q&A exchanges to display in chat


# ══════════════════════════════════════════════════════════════════════════
#  Streamlit Page Setup
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ChunkChat- A RAG PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .source-chunk {
        background-color: #f0f4f8;
        border-left: 4px solid #1f77b4;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 8px 0;
        font-size: 0.85rem;
    }
    .chat-user {
        background-color: #e3f2fd;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .chat-bot {
        background-color: #f5f5f5;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .status-box {
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  Session State Initialization
# ══════════════════════════════════════════════════════════════════════════
# Streamlit re-runs the entire script on every interaction.
# st.session_state persists data across re-runs within the same session.

def init_session_state():
    """Initialize all session state variables on first run."""

    if "chat_history" not in st.session_state:
        # Format: [{"role": "user"/"assistant", "content": "..."}]
        st.session_state.chat_history = []

    if "pdf_processed" not in st.session_state:
        # Flag: has a PDF been successfully processed?
        st.session_state.pdf_processed = False

    if "current_pdf_name" not in st.session_state:
        st.session_state.current_pdf_name = None

    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0

    if "collection" not in st.session_state:
        # Initialize ChromaDB connection once — reused across all interactions
        client = get_chroma_client()
        st.session_state.collection = get_or_create_collection(client)

        # Check if there's already data from a previous session
        info = get_collection_info(st.session_state.collection)
        if info["count"] > 0:
            st.session_state.pdf_processed = True
            st.session_state.chunk_count = info["count"]
            if info["sources"]:
                st.session_state.current_pdf_name = info["sources"][0]


init_session_state()


# ══════════════════════════════════════════════════════════════════════════
#  Sidebar — PDF Upload & Processing
# ══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📂 Document Upload")
    st.markdown("Upload a PDF to start asking questions about it.")

    uploaded_file = st.file_uploader(
        label="Choose a PDF file",
        type=["pdf"],
        help="Max file size: 200MB. Text-based PDFs work best."
    )

    # ── Process PDF Button ────────────────────────────────────────────────
    if uploaded_file is not None:
        st.info(f"📄 **{uploaded_file.name}** ({uploaded_file.size // 1024} KB)")

        process_btn = st.button(
            "⚙️ Process PDF",
            type="primary",
            use_container_width=True
        )

        if process_btn:
            # Save the uploaded file to disk
            pdf_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # ── Step 1: Extract Text ──────────────────────────────────────
            with st.spinner("📖 Extracting text from PDF..."):
                raw_text = load_pdf(pdf_path)

                if not raw_text:
                    st.error("❌ Could not extract text from this PDF. "
                             "Make sure it's a text-based PDF (not a scanned image).")
                    st.stop()

                metadata = get_pdf_metadata(pdf_path)
                st.success(f"✅ Text extracted: {len(raw_text):,} characters "
                           f"from {metadata['pages']} pages")

            # ── Step 2: Chunk Text ────────────────────────────────────────
            with st.spinner("✂️ Splitting text into chunks..."):
                chunks = chunk_text(raw_text, chunk_size=500, chunk_overlap=50)

                if not chunks:
                    st.error("❌ Text chunking produced no results.")
                    st.stop()

                stats = get_chunk_stats(chunks)
                st.success(f"✅ Created {stats['count']} chunks "
                           f"(avg {stats['avg_length']} chars each)")

            # ── Step 3: Generate Embeddings ───────────────────────────────
            with st.spinner(f"🧠 Generating embeddings for {len(chunks)} chunks... "
                            "(first run downloads model ~80MB)"):
                try:
                    embeddings = embed_texts(chunks)
                    st.success(f"✅ Generated {len(embeddings)} embeddings "
                               f"({len(embeddings[0])} dimensions each)")
                except Exception as e:
                    st.error(f"❌ Embedding generation failed: {e}")
                    st.stop()

            # ── Step 4: Store in ChromaDB ─────────────────────────────────
            with st.spinner("💾 Storing embeddings in ChromaDB..."):
                try:
                    # Clear old data when uploading a new document
                    clear_collection(st.session_state.collection)

                    count = store_embeddings(
                        chunks=chunks,
                        embeddings=embeddings,
                        doc_name=uploaded_file.name,
                        collection=st.session_state.collection
                    )

                    # Update session state
                    st.session_state.pdf_processed = True
                    st.session_state.current_pdf_name = uploaded_file.name
                    st.session_state.chunk_count = count
                    st.session_state.chat_history = []  # fresh chat for new doc

                    st.success(f"✅ Stored {count} chunks in ChromaDB")

                except Exception as e:
                    st.error(f"❌ Vector store error: {e}")
                    st.stop()

            st.balloons()
            st.success("🎉 PDF processed! Ask a question below.")

    # ── Current Document Status ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Status")

    if st.session_state.pdf_processed:
        st.success(f"✅ Active: **{st.session_state.current_pdf_name}**")
        st.metric("Chunks in DB", st.session_state.chunk_count)
        st.metric("Retrieval Top-K", TOP_K_CHUNKS)

        if st.button("🗑️ Clear Database", use_container_width=True):
            clear_collection(st.session_state.collection)
            st.session_state.pdf_processed = False
            st.session_state.current_pdf_name = None
            st.session_state.chunk_count = 0
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.warning("⚠️ No document loaded. Upload a PDF to begin.")

    # ── Groq API Key Status ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔑 API Configuration")

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        masked = f"{groq_key[:8]}...{groq_key[-4:]}"
        st.success(f"✅ Groq API Key: `{masked}`")
    else:
        st.error("❌ GROQ_API_KEY not set in .env file")
        st.code("GROQ_API_KEY=gsk_your_key_here", language="bash")

    st.markdown("---")
    st.markdown(
        "**Stack:** LangChain · ChromaDB · Sentence Transformers · Groq\n\n"
        "**Model:** all-MiniLM-L6-v2 · llama3-8b-8192"
    )


# ══════════════════════════════════════════════════════════════════════════
#  Main Panel — Chat Interface
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">📄 ChunkChat- A RAG PDF Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a PDF and ask questions — '
    'answers are grounded in your document.</div>',
    unsafe_allow_html=True
)

# ── Architecture Overview (expandable) ────────────────────────────────────
with st.expander("🏗️ How it works (RAG Architecture)", expanded=False):
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    RAG PIPELINE                                  │
    │                                                                  │
    │  INDEXING (one-time):                                            │
    │  PDF → Extract Text → Split Chunks → Embed → Store in ChromaDB  │
    │                                                                  │
    │  RETRIEVAL + GENERATION (per query):                             │
    │  Question → Embed Question → Similarity Search in ChromaDB       │
    │          → Top-3 Chunks → Groq LLM (llama3-8b) → Answer         │
    └─────────────────────────────────────────────────────────────────┘
    ```
    **Key concepts:**
    - **Chunking:** Split 50-page PDF into ~500-char pieces (context window limits)
    - **Embeddings:** Convert text to 384-dim vectors (similar text = nearby vectors)
    - **Cosine Similarity:** Find chunks most semantically similar to your question
    - **Grounded Generation:** LLM answers ONLY from retrieved chunks, reducing hallucinations
    """)

# ── Chat History Display ───────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation")

    # Display last MAX_HISTORY exchanges (each exchange = 2 messages: user + bot)
    recent = st.session_state.chat_history[-(MAX_HISTORY * 2):]

    for msg in recent:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(msg["content"])
                # Show source chunks if stored in the message
                if "sources" in msg and msg["sources"]:
                    with st.expander(
                        f"📚 Source Chunks Used ({len(msg['sources'])})",
                        expanded=False
                    ):
                        for i, source in enumerate(msg["sources"]):
                            st.markdown(
                                f'<div class="source-chunk">'
                                f'<strong>Chunk {i+1}:</strong> {source}'
                                f'</div>',
                                unsafe_allow_html=True
                            )

# ── Question Input ─────────────────────────────────────────────────────────
st.markdown("### ❓ Ask a Question")

# Disable input if no PDF is loaded
if not st.session_state.pdf_processed:
    st.info("👈 Please upload and process a PDF document first.")

question = st.chat_input(
    placeholder="e.g. What is the main topic of this document?",
    disabled=not st.session_state.pdf_processed
)

# ── RAG Query Pipeline ──────────────────────────────────────────────────────
if question and st.session_state.pdf_processed:
    # Display user's question immediately
    with st.chat_message("user"):
        st.write(question)

    # Store user message in history
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching document + generating answer..."):

            # ── Step A: Embed the Query ──────────────────────────────────
            try:
                query_vector = embed_query(question)
            except Exception as e:
                st.error(f"❌ Failed to embed question: {e}")
                st.stop()

            # ── Step B: Retrieve Similar Chunks ─────────────────────────
            # This is the "R" in RAG — retrieve relevant context
            try:
                results = query_similar_chunks(
                    query_embedding=query_vector,
                    collection=st.session_state.collection,
                    n_results=TOP_K_CHUNKS
                )
            except Exception as e:
                st.error(f"❌ Vector search failed: {e}")
                st.stop()

            # Extract the retrieved text chunks
            retrieved_chunks = results["documents"][0] if results["documents"] else []
            distances = results["distances"][0] if results["distances"] else []

            if not retrieved_chunks:
                st.warning("⚠️ No relevant chunks found. "
                           "Try rephrasing your question.")
                st.stop()

            # ── Step C: Generate Answer via Groq ────────────────────────
            # This is the "G" in RAG — generate a grounded answer
            answer = generate_answer(
                question=question,
                context_chunks=retrieved_chunks,
                chat_history=st.session_state.chat_history[:-1]  # exclude current Q
            )

        # Display the answer
        st.write(answer)

        # Show similarity scores alongside source chunks
        with st.expander(
            f"📚 Source Chunks Used ({len(retrieved_chunks)})",
            expanded=False
        ):
            for i, (chunk, dist) in enumerate(
                zip(retrieved_chunks, distances)
            ):
                # Convert distance to similarity score (0–1 scale)
                similarity = round(1 - dist, 3) if dist is not None else "N/A"
                st.markdown(
                    f'<div class="source-chunk">'
                    f'<strong>Chunk {i+1}</strong> '
                    f'<em>(similarity: {similarity})</em><br>{chunk}'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # Store assistant response in history (with sources for display)
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved_chunks
    })

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.8rem;'>"
    "RAG Chatbot · Built with LangChain, ChromaDB, Sentence Transformers & Groq"
    "</div>",
    unsafe_allow_html=True
)
