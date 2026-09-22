import streamlit as st
import os
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
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
        height: 100vh !important;
    }
    .stApp {
        height: 100vh;
        overflow: hidden;
        background: linear-gradient(180deg, #eef4ff 0%, #f7f9fc 220px, #f7f9fc 100%);
    }
    [data-testid="stMain"] {
        height: 100vh;
        overflow: hidden;
    }
    [data-testid="stMainBlockContainer"] {
        overflow: hidden;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 800px;
    }

    /* Header with logo */
    .app-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 14px;
        margin-bottom: 14px;
        border-bottom: 1px solid #dbe4f5;
    }
    .app-logo {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #3b82f6, #1e40af);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 800;
        font-size: 15px;
        flex-shrink: 0;
        box-shadow: 0 4px 10px rgba(30, 64, 175, 0.25);
    }
    .app-header-text h1 {
        margin: 0;
        font-size: 24px;
        color: #1e293b;
        font-weight: 700;
        line-height: 1.2;
    }
    .app-header-text p {
        margin: 2px 0 0 0;
        color: #64748b;
        font-size: 13px;
    }

    /* Chat container — the ONLY scrollable zone */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #dbe4f5;
        border-radius: 14px;
        background-color: #ffffff;
        overflow-y: auto !important;
    }

    .stChatMessage {
        border-radius: 14px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    [data-testid="stChatMessageContent"] {
        font-size: 15px;
        line-height: 1.5;
    }
    .stExpander {
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    [data-testid="stFileUploader"] {
        border-radius: 10px;
        border: 1px solid #c7d7f5;
    }

    /* Sidebar theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #24304a 100%);
        border-right: 1px solid #16213a;
    }
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: #2d3a52;
        border: 1px solid #3d4d6b;
    }

    /* Distinct button colors */
    [data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(1) button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(1) button:hover {
        background-color: #2563eb;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(2) button {
        background-color: transparent;
        color: #f87171;
        border: 1px solid #f87171;
        border-radius: 8px;
        font-weight: 600;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type(2) button:hover {
        background-color: #f87171;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def format_sources(chunks):
    """Group retrieved chunks by source file, with unique sorted page numbers."""
    grouped = {}
    for c in chunks:
        grouped.setdefault(c["source"], set()).add(c["page"])

    lines = []
    for source, pages in grouped.items():
        page_list = ", ".join(str(p) for p in sorted(pages))
        label = "page" if len(pages) == 1 else "pages"
        lines.append(f"{source} — {label} {page_list}")
    return lines


# ---- Header with logo ----
st.markdown("""
<div class="app-header">
    <div class="app-logo">RAG</div>
    <div class="app-header-text">
        <h1>Domain-Specific RAG Chatbot</h1>
        <p>Ask questions answered only from your uploaded documents — grounded, source-cited, no hallucinations.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "store" not in st.session_state:
    st.session_state.store = None

# Sidebar: upload + process
with st.sidebar:
    st.header("📁 Document Manager")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        st.write("Uploaded:")
        for f in uploaded_files:
            st.write(f"- {f.name}")

    if st.button("Process Documents", use_container_width=True):
        if not uploaded_files:
            st.error("Please upload at least one PDF first.")
        else:
            os.makedirs("documents", exist_ok=True)
            for f in uploaded_files:
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

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Load existing vector store if one exists and hasn't been loaded this session
if st.session_state.store is None and os.path.exists("vector_store/saved_index/index.faiss"):
    store = VectorStore()
    store.load()
    st.session_state.store = store

# ---- Scrollable middle chat zone (only this scrolls, not the page) ----
if st.session_state.store is None:
    st.info("Upload PDFs and click **Process Documents** in the sidebar to get started.")
else:
    chat_container = st.container(height=480, border=True)

    with chat_container:
        for q, a, chunks in st.session_state.history:
            with st.chat_message("user", avatar="🧑"):
                st.write(q)

            with st.chat_message("assistant", avatar="📄"):
                st.write(a)
                if "could not find this information" not in a.lower():
                    with st.expander("📄 Sources used"):
                        for line in format_sources(chunks):
                            st.caption(line)

    # Native chat_input docks to the bottom of the viewport
    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:
        with chat_container:
            with st.chat_message("user", avatar="🧑"):
                st.write(user_question)

            with st.chat_message("assistant", avatar="📄"):
                with st.spinner("Thinking..."):
                    chunks = st.session_state.store.search(user_question, top_k=4)
                    answer = generate_answer(user_question, chunks)
                st.write(answer)
                if "could not find this information" not in answer.lower():
                    with st.expander("📄 Sources used"):
                        for line in format_sources(chunks):
                            st.caption(line)

        st.session_state.history.append((user_question, answer, chunks))
        st.rerun()

