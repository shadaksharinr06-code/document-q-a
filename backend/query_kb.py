import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

# 1. Initialize Embeddings & Chroma Retriever
def initialize_retriever(persist_directory=DB_DIR, k=4):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

# 2. Strict Prompt Template with Dual Response Format
PROMPT_TEMPLATE = """You are a precise Document Q&A Assistant. For "document_grounded_answer", answer the user's question using ONLY the provided context below.
- If the answer is not explicitly contained in the context, reply: "I cannot find this information in the provided documents."
- Do not make assumptions or bring in outside knowledge.
- Cite the document name or section if available.

Your response MUST be a valid JSON object matching this schema:
{{
  "document_grounded_answer": "...",
  "citations": ["Document: <filename>, Page X"],
  "supplemental_knowledge_answer": "..."
}}

Context:
{context}

Question:
{question}

Instructions:
1. "document_grounded_answer": Answer using ONLY the provided context above (or reply "I cannot find this information in the provided documents.").
2. "citations": Array of citation tags for the chunks used.
3. "supplemental_knowledge_answer": Broader context, industry standards, or world knowledge outside the document."""

def query_knowledge_base(question: str):
    if not os.path.exists(DB_DIR):
        print(f"Error: Chroma database directory '{DB_DIR}' not found. Run train_kb.py first.")
        return

    # Use the retriever as defined
    retriever = initialize_retriever(persist_directory=DB_DIR, k=4)
    docs = retriever.invoke(question)

    if not docs:
        print("\n[FROM YOUR DOCUMENT]")
        print("I cannot find this information in the provided documents.")
        return

    # Format retrieved document context
    context_chunks = []
    for idx, doc in enumerate(docs):
        src = os.path.basename(doc.metadata.get("source", "Document"))
        page = doc.metadata.get("page", 0) + 1
        context_chunks.append(
            f"--- CHUNK {idx + 1} [Document: {src}, Page {page}] ---\n{doc.page_content}\n"
        )
    formatted_context = "\n".join(context_chunks)

    # Build prompt
    prompt = PROMPT_TEMPLATE.format(context=formatted_context, question=question)

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a precise Document Q&A Assistant."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        data = json.loads(completion.choices[0].message.content)
        print("\n================== [FROM YOUR DOCUMENT] ==================")
        print(data.get("document_grounded_answer"))
        citations = data.get("citations", [])
        if citations:
            print(f"Citations: {', '.join(citations)}")
        print("\n============= [SUPPLEMENTAL INSIGHTS (AI)] =============")
        print(data.get("supplemental_knowledge_answer"))
        print("========================================================\n")
    else:
        print("\n--- Retrieved Context from ChromaDB (k=4) ---")
        for chunk in context_chunks:
            print(chunk)
        print("Set OPENAI_API_KEY to generate live dual-response GPT-4o synthesis.")

if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else input("Enter your question: ")
    query_knowledge_base(query)
