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
    .stApp {
        background-color: #f7f9fc;
    }
    h1 {
        color: #1e293b;
        font-weight: 700;
    }
    [data-testid="stSidebar"] {
        background-color: #eef2f9;
        border-right: 1px solid #dde3ee;
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
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
    }
    .stExpander {
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    [data-testid="stFileUploader"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 Domain-Specific RAG Chatbot")
st.caption("Ask questions answered only from your uploaded documents — grounded, source-cited, no hallucinations.")

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

# Chat interface
st.divider()

if st.session_state.store is None:
    st.info("Upload PDFs and click **Process Documents** in the sidebar to get started.")
else:
    # Display chat history using chat bubbles
    for q, a, chunks in st.session_state.history:
        with st.chat_message("user", avatar="🧑"):
            st.write(q)

        with st.chat_message("assistant", avatar="📄"):
            st.write(a)
            if "could not find this information" not in a.lower():
                with st.expander("📄 Sources used"):
                    for c in chunks:
                        st.caption(f"{c['source']} — page {c['page']}")

    # Chat input pinned at the bottom
    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:
        with st.chat_message("user", avatar="🧑"):
            st.write(user_question)

        with st.chat_message("assistant", avatar="📄"):
            with st.spinner("Thinking..."):
                chunks = st.session_state.store.search(user_question, top_k=4)
                answer = generate_answer(user_question, chunks)
            st.write(answer)
            if "could not find this information" not in answer.lower():
                with st.expander("📄 Sources used"):
                    for c in chunks:
                        st.caption(f"{c['source']} — page {c['page']}")

        st.session_state.history.append((user_question, answer, chunks))