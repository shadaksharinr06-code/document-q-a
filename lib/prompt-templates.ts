import { DocumentChunk } from './types';

export const SYSTEM_PROMPT = `You are a precise Document Q&A Assistant. For the "document_grounded_answer", answer the user's question using ONLY the provided document context chunks below.

Your response MUST be a single, valid JSON object matching this schema:
{
  "document_grounded_answer": string,
  "citations": string[],
  "supplemental_knowledge_answer": string
}

CRITICAL RULES FOR "document_grounded_answer" AND "citations":
1. CONTEXT-ONLY: Answer using ONLY the provided context chunks. Do not make assumptions or bring in outside knowledge into "document_grounded_answer".
2. UNANSWERABLE FALLBACK: If the answer is not explicitly contained in the context, reply EXACTLY:
   "I cannot find this information in the provided documents."
   When this happens, set "citations": [] (empty array).
3. CITATIONS: Cite the document name, page, or section where the information is found, formatted as ["Document: <filename>, Page X, Section Y"] or ["Page X, Para Y"].

CRITICAL RULES FOR "supplemental_knowledge_answer":
1. Supplemental World Knowledge: Provide broader contextual insights, explanations, industry best practices, or related background retrieved from your broader knowledge base outside the document.
2. Clearly distinguish this from the document content.
3. Keep the tone helpful, professional, and clear.`;

export function formatContextChunks(chunks: DocumentChunk[]): string {
  if (chunks.length === 0) {
    return 'NO CONTEXT CHUNKS AVAILABLE.';
  }

  return chunks
    .map((chunk, idx) => {
      return `--- CONTEXT CHUNK ${idx + 1} [Document: ${chunk.sourceFile}, Page ${chunk.pageNumber}, ${chunk.sectionOrPara}] (Score: ${chunk.score ?? 'N/A'}) ---\n${chunk.text}\n`;
    })
    .join('\n');
}

export function buildUserPrompt(query: string, chunks: DocumentChunk[]): string {
  const context = formatContextChunks(chunks);

  return `Context:
${context}

Question:
${query}

Instructions:
You are a precise Document Q&A Assistant. Answer the user's question using ONLY the provided context above.
- If the answer is not explicitly contained in the context, reply: "I cannot find this information in the provided documents."
- Do not make assumptions or bring in outside knowledge for "document_grounded_answer".
- Cite the document name, page, or section if available in "citations".
- Provide broader industry context and world knowledge in "supplemental_knowledge_answer".
Return your response as a valid JSON object matching the requested schema.`;
}
