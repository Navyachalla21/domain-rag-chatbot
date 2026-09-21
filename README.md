# 📄 Domain-Specific RAG Chatbot

**Live App:** [https://domain-rag-chatbotgi-jajxu6e28othk9ehy6ys7j.streamlit.app](https://domain-rag-chatbotgi-jajxu6e28othk9ehy6ys7j.streamlit.app)

A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly from user-uploaded PDF documents. It retrieves the most relevant passages from the uploaded content and generates grounded answers — citing the exact source document and page number, and refusing to answer when the information isn't present in the documents.

## Problem Statement

Large documents are difficult to search manually — a user may need to read through many pages just to find one answer. This project solves that by letting users upload PDFs and ask questions in natural language, getting back answers grounded only in that content, with clear source attribution.

## Tools Used

| Component | Tool |
|---|---|
| Programming language | Python |
| PDF text extraction | pypdf |
| Text splitting | LangChain text splitters |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector database | FAISS |
| Language model | Google Gemini (`gemini-3.6-flash`) |
| User interface | Streamlit |
| Environment variables | python-dotenv |
| Deployment | Streamlit Community Cloud |
| Version control | Git and GitHub |

## How It Works

1. The user uploads one or more PDF files through the sidebar.
2. Each page's text is extracted using `pypdf`, tagged with its source filename and page number.
3. The extracted text is split into smaller overlapping chunks (chunk size ~800 characters, overlap ~120 characters) so each chunk captures a focused idea.
4. Each chunk is converted into a numerical embedding using the `all-MiniLM-L6-v2` model and stored in a FAISS vector index, alongside its source/page metadata.
5. When the user asks a question, the question is embedded the same way, and FAISS retrieves the most similar chunks.
6. The retrieved chunks and the question are sent to Gemini with a strict prompt: answer only from the given context, and explicitly say "I could not find this information in the uploaded documents" if the answer isn't present.
7. The answer is displayed in a chat interface, with the exact source document(s) and page number(s) shown alongside it.

## Architecture

![RAG Chatbot Architecture](rag_architecture.png)

## Project Structure

domain_rag_chatbot/
├── app.py # Streamlit interface
├── document_loader.py # PDF text extraction (Module 2)
├── rag_pipeline.py # Text chunking (Module 3)
├── vector_store.py # Embeddings + FAISS storage/retrieval (Modules 4-5)
├── prompt.py # Answer generation with Gemini (Module 6)
├── documents/ # Uploaded/sample PDFs
├── vector_store/saved_index/ # Saved FAISS index (generated, not committed)
├── tests/test_questions.csv # Testing sheet
├── requirements.txt
└── README.md


## Setup and Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Navyachalla21/domain-rag-chatbot.git
cd domain-rag-chatbot
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Gemini API key**
Create a `.env` file in the project root:

GOOGLE_API_KEY=your_key_here

Get a free key from [Google AI Studio](https://aistudio.google.com).

**5. Run the app**
```bash
streamlit run app.py
```

**6. Use it**
- Upload a PDF (or multiple) via the sidebar
- Click **Process Documents**
- Ask questions in the chat box — answers are grounded only in the uploaded content, with sources shown below each answer

## Testing

The chatbot was tested with 15 questions covering three categories:
- **In-scope questions** — directly answerable from the uploaded PDF content
- **Edge-case questions** — reasonable questions that may not be explicitly covered
- **Out-of-scope questions** — unrelated to the uploaded documents, to confirm the refusal guardrail works correctly

Full results are in [`tests/test_questions.csv`](tests/test_questions.csv).

## Limitations

- Answer quality depends entirely on what's in the uploaded PDF — the chatbot cannot answer from general knowledge.
- Scanned/image-only PDF pages without extractable text are skipped.
- Retrieval occasionally misses relevant passages for loosely-worded questions, correctly triggering a refusal even when related information technically exists elsewhere in the document.

## Responsible AI Notes

- The chatbot only answers from retrieved document context and explicitly refuses when information isn't found, rather than inventing an answer.
- API keys are never committed to the repository (`.env` is git-ignored) and are stored securely as encrypted secrets on Streamlit Cloud.

