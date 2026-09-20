from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)    

SYSTEM_PROMPT = """You are a document question-answering assistant.

Answer only from the supplied context. If the answer is not available in the context, say:
"I could not find this information in the uploaded documents."

Do not invent facts. Do not mention sources or page numbers in your answer — 
just give the answer itself."""

def generate_answer(question, retrieved_chunks):
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.append(
            f"[Source: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    full_prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = llm.invoke(full_prompt)
        content = response.content
        if isinstance(content, list):
            answer = "\n".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            answer = content
        return answer
    except Exception as e:
        return f"Something went wrong generating the answer: {e}"


if __name__ == "__main__":
    from vector_store import VectorStore

    store = VectorStore()
    store.load()

    question = "What is a list comprehension?"
    chunks = store.search(question, top_k=3)
    answer = generate_answer(question, chunks)

    print(f"Q: {question}\n")
    print(f"A: {answer}")
