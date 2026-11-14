import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

SYSTEM_PROMPT = """You are a precise Document Q&A Assistant. For "document_grounded_answer", answer the user's question using ONLY the provided document context below.

Your response MUST be a single, valid JSON object matching this schema:
{
  "document_grounded_answer": "...",
  "citations": ["Document: <filename>, Page X"],
  "supplemental_knowledge_answer": "..."
}

CRITICAL RULES FOR "document_grounded_answer" AND "citations":
1. CONTEXT-ONLY: Answer using ONLY the provided context. Do not make assumptions or bring in outside knowledge into "document_grounded_answer".
2. UNANSWERABLE FALLBACK: If the answer is not explicitly contained in the context, reply EXACTLY:
   "I cannot find this information in the provided documents."
   When this happens, set "citations": [] (empty array).
3. CITATIONS: Cite the document name, page, or section if available.

CRITICAL RULES FOR "supplemental_knowledge_answer":
1. Provide broader contextual insights, industry best practices, explanations, or related background retrieved from your broader knowledge base outside the document.
2. Keep the tone helpful, professional, and clear."""

def query_knowledge_base(question: str, top_k: int = 4):
    if not os.path.exists(DB_DIR):
        print(f"Error: Chroma database not found at {DB_DIR}. Run train_kb.py first.")
        return

    # 1. Load Chroma Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # 2. Retrieve top-k documents with similarity scores
    results = vectorstore.similarity_search_with_relevance_scores(question, k=top_k)
    if not results:
        print("I cannot find this information in the provided documents.")
        return

    # Format context blocks
    context_blocks = []
    for idx, (doc, score) in enumerate(results):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 0) + 1
        context_blocks.append(
            f"--- CONTEXT CHUNK {idx + 1} [Document: {os.path.basename(source)}, Page {page}] (Relevance: {score:.3f}) ---\n{doc.page_content}\n"
        )

    context_str = "\n".join(context_blocks)

    user_prompt = f"""Context:
{context_str}

Question:
{question}

Instructions:
You are a precise Document Q&A Assistant. Answer the user's question using ONLY the provided context above.
- If the answer is not explicitly contained in the context, reply: "I cannot find this information in the provided documents."
- Do not make assumptions or bring in outside knowledge for "document_grounded_answer".
- Cite the document name or section if available in "citations".
- Provide broader industry context and world knowledge in "supplemental_knowledge_answer"."""

    # 3. Call LLM for Dual-Response Synthesis
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
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
        # Local extractive display if no OpenAI key is set
        print("\nContext Chunks Retrieved from ChromaDB:")
        for block in context_blocks:
            print(block)
        print("\nSet OPENAI_API_KEY to generate live dual-response GPT-4o synthesis.")

if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else input("Enter your question: ")
    query_knowledge_base(query)
