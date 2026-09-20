import os
from pypdf import PdfReader


def load_documents(folder_path):
    """
    Reads all PDF files in a folder and extracts text page by page.
    Returns a list of dicts: {"text": ..., "source": filename, "page": page_number}
    """
    documents = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(folder_path, filename)
        reader = PdfReader(filepath)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            # Skip empty pages safely (scanned pages with no extractable text, blank pages, etc.)
            if not text or not text.strip():
                continue

            documents.append({
                "text": text,
                "source": filename,
                "page": page_number
            })

    return documents


if __name__ == "__main__":
    docs = load_documents("documents")
    print(f"Extracted {len(docs)} pages of text.")
    if docs:
        print("Sample entry:")
        print(docs[0])