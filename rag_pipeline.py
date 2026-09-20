from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=800, chunk_overlap=120):
    """
    Takes the list of page dicts from document_loader and splits each
    page's text into smaller overlapping chunks, preserving source/page metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for doc in documents:
        split_texts = splitter.split_text(doc["text"])
        for chunk_text in split_texts:
            chunks.append({
                "text": chunk_text,
                "source": doc["source"],
                "page": doc["page"]
            })

    return chunks


if __name__ == "__main__":
    from document_loader import load_documents

    docs = load_documents("documents")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} pages.")
    if chunks:
        print("Sample chunk:")
        print(chunks[0])