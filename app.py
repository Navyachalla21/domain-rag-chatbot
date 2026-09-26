import streamlit as st
import os
import re
from document_loader import load_documents
from rag_pipeline import chunk_documents
from vector_store import VectorStore
from prompt import generate_answer

st.set_page_config(
    page_title="Domain RAG Chatbot",
    page_icon="📄",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 820px;
    }

    .app-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 14px 0 16px 0;
        margin-bottom: 18px;
        border-bottom: 1px solid #e2e8f0;
        position: sticky;
        top: 0;
        background-color: #f8fafc;
        z-index: 100;
    }
    .app-logo {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: linear-gradient(135deg, #4f46e5, #4338ca);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 800;
        font-size: 16px;
        flex-shrink: 0;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
    }
    .app-header-text h1 {
        margin: 0;
        font-size: 26px;
        color: #0f172a;
        font-weight: 700;
        line-height: 1.2;
    }
    .app-header-text p {
        margin: 3px 0 0 0;
        color: #64748b;
        font-size: 13.5px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        background-color: #ffffff;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }

    .stChatMessage {
        border-radius: 14px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    [data-testid="stChatMessageContent"] {
        font-size: 15px;
        line-height: 1.6;
        color: #1e293b;
    }
    .stExpander {
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-top: 4px;
    }
    [data-testid="stFileUploader"] {
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        background-color: #ffffff;
    }

    [data-testid="stSidebar"] {
        background-color: #f1f5f9;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] h2 {
        color: #0f172a;
        font-weight: 700;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: #334155 !important;
    }

    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #4f46e5;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background-color: #4338ca;
        color: #ffffff;
    }

    div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #ffffff;
        color: #dc2626;
        border: 1.5px solid #dc2626;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background-color: #dc2626;
        color: #ffffff;
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

MAX_FILE_SIZE_MB = 50


def format_sources(chunks):
    grouped = {}
    for c in chunks:
        grouped.setdefault(c["source"], set()).add(c["page"])
    lines = []
    for source, pages in grouped.items():
        page_list = ", ".join(str(p) for p in sorted(pages))
        label = "page" if len(pages) == 1 else "pages"
        lines.append(f"{source} — {label} {page_list}")
    return lines


def strip_inline_sources(text):
    """Remove inline [Source: ...] mentions from the model's answer text,
    since sources are already shown cleanly in the expander below."""
    return re.sub(r"\[?Source:.*?\]?(\n|$)", "", text, flags=re.IGNORECASE).strip()


st.markdown("""
<div class="app-header">
    <div class="app-logo">RAG</div>
    <div class="app-header-text">
        <h1>Domain-Specific RAG Chatbot</h1>
        <p>Ask questions answered only from your uploaded documents — grounded, source-cited, no hallucinations.</p>
    </div>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "store" not in st.session_state:
    st.session_state.store = None

with st.sidebar:
    st.header("📁 Document Manager")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        st.write("Uploaded:")
        oversized = []
        for f in uploaded_files:
            size_mb = f.size / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                oversized.append(f.name)
            st.write(f"- {f.name} ({size_mb:.1f} MB)")

        if oversized:
            st.error(f"These files exceed the {MAX_FILE_SIZE_MB}MB limit and won't be processed: {', '.join(oversized)}")

    if st.button("Process Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.error("Please upload at least one PDF first.")
        else:
            valid_files = [f for f in uploaded_files if f.size / (1024 * 1024) <= MAX_FILE_SIZE_MB]
            if not valid_files:
                st.error("No valid files to process — all uploads exceed the size limit.")
            else:
                os.makedirs("documents", exist_ok=True)
                for f in valid_files:
                    with open(os.path.join("documents", f.name), "wb") as out:
                        out.write(f.getbuffer())

                with st.spinner("Extracting, chunking, and embedding documents..."):
                    docs = load_documents("documents")
                    chunks = chunk_documents(docs)
                    store = VectorStore()
                    store.build(chunks)
                    store.save()
                    st.session_state.store = store

                st.success(f"Processed {len(docs)} pages into {len(chunks)} chunks.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            st.session_state.history = []
            st.rerun()
    with col2:
        if st.button("📄 Clear Docs", use_container_width=True, type="secondary"):
            st.session_state.store = None
            st.session_state.history = []
            index_path = "vector_store/saved_index/index.faiss"
            chunks_path = "vector_store/saved_index/chunks.pkl"
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(chunks_path):
                os.remove(chunks_path)
            if os.path.exists("documents"):
                for f in os.listdir("documents"):
                    os.remove(os.path.join("documents", f))
            st.success("Documents and index cleared. Upload new PDFs to start fresh.")
            st.rerun()

if st.session_state.store is None and os.path.exists("vector_store/saved_index/index.faiss"):
    store = VectorStore()
    store.load()
    st.session_state.store = store

if st.session_state.store is None:
    st.info("Upload PDFs and click **Process Documents** in the sidebar to get started.")
else:
    chat_container = st.container(height=480, border=True)

    with chat_container:
        for q, a, chunks in st.session_state.history:
            with st.chat_message("user", avatar="🧑"):
                st.write(q)
            with st.chat_message("assistant", avatar="📄"):
                st.write(strip_inline_sources(a))
                if "could not find this information" not in a.lower():
                    with st.expander("📄 Sources used"):
                        for line in format_sources(chunks):
                            st.caption(line)

    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:
        with chat_container:
            with st.chat_message("user", avatar="🧑"):
                st.write(user_question)
            with st.chat_message("assistant", avatar="📄"):
                with st.spinner("Thinking..."):
                    chunks = st.session_state.store.search(user_question, top_k=4)
                    answer = generate_answer(user_question, chunks)
                st.write(strip_inline_sources(answer))
                if "could not find this information" not in answer.lower():
                    with st.expander("📄 Sources used"):
                        for line in format_sources(chunks):
                            st.caption(line)

        st.session_state.history.append((user_question, answer, chunks))
        st.rerun()