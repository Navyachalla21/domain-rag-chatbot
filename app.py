import streamlit as st
import os
from document_loader import load_documents
from rag_pipeline import chunk_documents
from vector_store import VectorStore
from prompt import generate_answer

st.set_page_config(page_title="Domain RAG Chatbot", page_icon="📄")
st.title("📄 Domain-Specific RAG Chatbot")
st.write("Upload PDFs, then ask questions answered only from their content.")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "store" not in st.session_state:
    st.session_state.store = None

# Sidebar: upload + process
with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        st.write("Uploaded:")
        for f in uploaded_files:
            st.write(f"- {f.name}")

    if st.button("Process Documents"):
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

    if st.button("Clear Chat"):
        st.session_state.history = []

# Load existing vector store if one exists and hasn't been loaded this session
if st.session_state.store is None and os.path.exists("vector_store/saved_index/index.faiss"):
    store = VectorStore()
    store.load()
    st.session_state.store = store

# Chat interface
with st.form(key="question_form", clear_on_submit=True):
    user_question = st.text_input("Ask a question about your documents:")
    submitted = st.form_submit_button("Ask")

if submitted and user_question:
    if st.session_state.store is None:
        st.error("Please upload and process documents first.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            chunks = st.session_state.store.search(user_question, top_k=4)
            answer = generate_answer(user_question, chunks)
            st.session_state.history.append((user_question, answer, chunks))

# Display chat history, most recent first
for q, a, chunks in reversed(st.session_state.history):
    st.write(f"**Q: {q}**")
    st.write(a)
    with st.expander("Sources used"):
        for c in chunks:
            st.write(f"- {c['source']}, page {c['page']}")
    st.write("---")

