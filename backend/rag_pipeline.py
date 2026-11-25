import os
import io
import math
import re
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

from pypdf import PdfReader
import docx

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    HAS_CHROMA_HF = True
except Exception:
    HAS_CHROMA_HF = False

from openai import OpenAI
from models import DualRAGResponse, ChunkItem, DocumentMetadata

SYSTEM_PROMPT = """You are a precise Document Q&A Assistant. For "document_grounded_answer", answer the user's question using ONLY the provided document context below.
- If the answer is not explicitly contained in the context, reply: "I cannot find this information in the provided documents."
- Do not make assumptions or bring in outside knowledge.
- Cite the document name or section if available.

Your response MUST be a valid JSON object matching this schema:
{
  "document_grounded_answer": "...",
  "citations": ["Document: <filename>, Page X"],
  "supplemental_knowledge_answer": "..."
}

CRITICAL RULES FOR "document_grounded_answer" AND "citations":
1. CONTEXT-ONLY: Answer using ONLY the provided context chunks. Do not make assumptions or bring in outside knowledge into "document_grounded_answer".
2. UNANSWERABLE FALLBACK: If the answer is not explicitly contained in the context, reply EXACTLY:
   "I cannot find this information in the provided documents."
   When this happens, set "citations": [] (empty array).
3. CITATIONS: Cite the document name, page, or section if available.

CRITICAL RULES FOR "supplemental_knowledge_answer":
1. Provide question-tailored extra information (definitions, background, real-world examples, best practices, or implications) specifically addressing the user's question outside the document.
2. Avoid generic boilerplate; make every point relevant to the exact question asked.
3. Keep the tone helpful, professional, and clear."""

class DocumentRAGPipeline:
    def __init__(self, session_id: str, api_key: Optional[str] = None):
        self.session_id = session_id
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.chunks: List[ChunkItem] = []
        self.metadata: Optional[DocumentMetadata] = None
        self.vectorstore = None
        self.retriever = None

    def parse_document(self, file_bytes: bytes, filename: str) -> Tuple[List[Dict[str, Any]], int, bool]:
        ext = filename.split(".")[-1].lower()
        pages = []
        is_scanned_warning = False

        if ext == "pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            total_pages = len(reader.pages)
            total_chars = 0
            for p_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                total_chars += len(text.strip())
                pages.append({"page_number": p_idx + 1, "text": text.strip()})
            if total_chars < 50:
                is_scanned_warning = True

        elif ext == "docx":
            doc = docx.Document(io.BytesIO(file_bytes))
            full_text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            paragraphs = full_text.split("\n\n")
            curr_page = 1
            buffer = []
            curr_len = 0
            for para in paragraphs:
                if curr_len + len(para) > 2500 and buffer:
                    pages.append({"page_number": curr_page, "text": "\n\n".join(buffer)})
                    curr_page += 1
                    buffer = [para]
                    curr_len = len(para)
                else:
                    buffer.append(para)
                    curr_len += len(para)
            if buffer:
                pages.append({"page_number": curr_page, "text": "\n\n".join(buffer)})
            total_pages = len(pages) or 1
            if len(full_text.strip()) < 30:
                is_scanned_warning = True

        elif ext == "txt":
            raw_text = file_bytes.decode("utf-8", errors="ignore")
            sections = raw_text.split("\n\n")
            curr_page = 1
            buffer = []
            curr_len = 0
            for sec in sections:
                if curr_len + len(sec) > 2000 and buffer:
                    pages.append({"page_number": curr_page, "text": "\n\n".join(buffer)})
                    curr_page += 1
                    buffer = [sec]
                    curr_len = len(sec)
                else:
                    buffer.append(sec)
                    curr_len += len(sec)
            if buffer:
                pages.append({"page_number": curr_page, "text": "\n\n".join(buffer)})
            total_pages = len(pages) or 1
            if len(raw_text.strip()) < 20:
                is_scanned_warning = True

        else:
            raise ValueError(f"Unsupported format: .{ext}")

        return pages, total_pages, is_scanned_warning

    def chunk_and_index(self, file_bytes: bytes, filename: str) -> DocumentMetadata:
        pages, total_pages, is_scanned = self.parse_document(file_bytes, filename)
        
        chunk_items: List[ChunkItem] = []
        chk_count = 1

        for page in pages:
            page_text = page["text"]
            if not page_text:
                continue
            paras = [p.strip() for p in page_text.split("\n\n") if p.strip()]
            for p_idx, para in enumerate(paras):
                # Simple recursive chunk split with 800 chars and 150 overlap
                step = 650
                for start in range(0, max(1, len(para)), step):
                    sub_chk = para[start:start+800].strip()
                    if not sub_chk:
                        continue
                    label = f"Para {p_idx + 1}"
                    chk_id = f"chk_{chk_count}"
                    chunk_items.append(
                        ChunkItem(
                            id=chk_id,
                            text=sub_chk,
                            page_number=page["page_number"],
                            section_or_para=label,
                            source_file=filename
                        )
                    )
                    chk_count += 1

        if not chunk_items:
            raise ValueError("No extractable text found in document.")

        self.chunks = chunk_items

        # Initialize trained model (HuggingFace + Chroma) if available
        if HAS_CHROMA_HF:
            try:
                embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
                texts = [c.text for c in chunk_items]
                metadatas = [
                    {
                        "source": c.source_file,
                        "page": c.page_number - 1,
                        "section": c.section_or_para,
                        "id": c.id
                    }
                    for c in chunk_items
                ]
                self.vectorstore = Chroma.from_texts(
                    texts=texts,
                    embedding=embeddings,
                    metadatas=metadatas
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
            except Exception as e:
                print("HuggingFace/Chroma initialization notice:", e)

        self.metadata = DocumentMetadata(
            id=self.session_id,
            name=filename,
            size=len(file_bytes),
            type=filename.split(".")[-1],
            page_count=total_pages,
            chunk_count=len(chunk_items),
            uploaded_at=datetime.utcnow().isoformat(),
            is_scanned_warning=is_scanned,
            status="ready"
        )
        return self.metadata

    def query(self, question: str, top_k: int = 4) -> Tuple[DualRAGResponse, List[ChunkItem]]:
        if not self.chunks:
            raise ValueError("Document not indexed. Upload a document first.")

        retrieved_chunks: List[ChunkItem] = []

        # 1. Retrieve via Chroma retriever if available
        if self.retriever:
            try:
                docs = self.retriever.invoke(question)
                for d in docs:
                    meta = d.metadata
                    retrieved_chunks.append(
                        ChunkItem(
                            id=meta.get("id", "chk"),
                            text=d.page_content,
                            page_number=meta.get("page", 0) + 1,
                            section_or_para=meta.get("section", "Section"),
                            source_file=meta.get("source", self.metadata.name if self.metadata else "")
                        )
                    )
            except Exception as e:
                print("Retriever invoke fallback:", e)

        # 2. Keyword BM25 fallback retrieval if retriever didn't populate
        if not retrieved_chunks:
            q_terms = set(re.findall(r'\w+', question.lower()))
            scored = []
            for c in self.chunks:
                c_terms = set(re.findall(r'\w+', c.text.lower()))
                overlap = len(q_terms.intersection(c_terms))
                scored.append((overlap, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            retrieved_chunks = [item[1] for item in scored[:top_k]]

        # Format context chunks
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks):
            context_blocks.append(
                f"--- CONTEXT CHUNK {idx + 1} [Document: {chunk.source_file}, Page {chunk.page_number}, {chunk.section_or_para}] ---\n{chunk.text}\n"
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
- Cite the document name, page, or section if available in "citations".
- Provide broader industry context and world knowledge in "supplemental_knowledge_answer"."""

        # 3. LLM Generation
        if self.api_key and not self.api_key.startswith("sk-dummy"):
            client = OpenAI(api_key=self.api_key)
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            import json
            data = json.loads(completion.choices[0].message.content)
            parsed_response = DualRAGResponse(
                document_grounded_answer=data.get("document_grounded_answer", "I cannot find this information in the provided documents."),
                citations=data.get("citations", []),
                supplemental_knowledge_answer=data.get("supplemental_knowledge_answer", "")
            )
            return parsed_response, retrieved_chunks
        else:
            # Fallback offline answer using top retrieved chunk
            top_chunk = retrieved_chunks[0] if retrieved_chunks else None
            doc_name = self.metadata.name if self.metadata else "Document"
            if top_chunk and any(w in top_chunk.text.lower() for w in re.findall(r'\w+', question.lower()) if len(w) > 3):
                return DualRAGResponse(
                    document_grounded_answer=f"From {doc_name} (Page {top_chunk.page_number}, {top_chunk.section_or_para}):\n\n\"{top_chunk.text}\"",
                    citations=[f"Document: {doc_name}, Page {top_chunk.page_number}"],
                    supplemental_knowledge_answer=f"Context regarding '{question}': Standard industry practices suggest validating source requirements against established operational benchmarks."
                ), retrieved_chunks
            else:
                return DualRAGResponse(
                    document_grounded_answer="I cannot find this information in the provided documents.",
                    citations=[],
                    supplemental_knowledge_answer=f"While '{doc_name}' does not contain details regarding '{question}', industry knowledge indicates this topic relates to broader domain guidelines and established frameworks."
                ), retrieved_chunks
