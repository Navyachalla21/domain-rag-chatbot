from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os


class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []  # keeps chunk text + metadata, same order as vectors in the index

    def build(self, chunks):
        """
        chunks: list of dicts with 'text', 'source', 'page'
        """
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def save(self, folder="vector_store/saved_index"):
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, folder="vector_store/saved_index"):
        self.index = faiss.read_index(os.path.join(folder, "index.faiss"))
        with open(os.path.join(folder, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)

    def search(self, query, top_k=4):
        query_vector = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results


if __name__ == "__main__":
    from document_loader import load_documents
    from rag_pipeline import chunk_documents

    docs = load_documents("documents")
    chunks = chunk_documents(docs)

    store = VectorStore()
    store.build(chunks)
    store.save()
    print(f"Vector store built and saved with {len(chunks)} chunks.")

    # Quick sanity test
    results = store.search("What is a list comprehension?", top_k=3)
    print("\nTop matches for test query:")
    for r in results:
        print(f"- {r['source']} (page {r['page']}): {r['text'][:100]}...")