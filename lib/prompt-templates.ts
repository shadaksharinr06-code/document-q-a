import { DocumentChunk } from './types';

export const SYSTEM_PROMPT = `You are a specialized Dual-Response Document Assistant. You analyze user questions using retrieved excerpts from their uploaded document while simultaneously delivering broader industry or world knowledge.

Your response MUST be a single, valid JSON object matching this schema:
{
  "document_grounded_answer": string,
  "citations": string[],
  "supplemental_knowledge_answer": string
}

CRITICAL RULES FOR "document_grounded_answer" AND "citations":
1. ZERO HALLUCINATION POLICY: The "document_grounded_answer" MUST be derived EXCLUSIVELY and FACTUALLY from the provided Context Chunks. Do not introduce any outside facts, assumptions, or deductions not stated in the text.
2. CITATIONS: Include every specific chunk citation that supported the answer, formatted as ["Page X, Para Y"] or ["Page X, Section Y"].
3. UNANSWERABLE FALLBACK: If the provided Context Chunks do NOT contain the answer, or if the question asks for details not present in the document, you MUST output EXACTLY:
   "This specific information was not found in the uploaded document."
   When this happens, set "citations": [] (empty array).

CRITICAL RULES FOR "supplemental_knowledge_answer":
1. Provide rich, expert-level world knowledge, broader industry context, theoretical background, real-world best practices, or relevant comparisons beyond what is in the document.
2. This section is explicitly NOT restricted to the document. It should help the user understand the wider picture, common pitfalls, or strategic implications related to their inquiry.
3. Keep the tone helpful, professional, and clear.`;

export function formatContextChunks(chunks: DocumentChunk[]): string {
  if (chunks.length === 0) {
    return 'NO CONTEXT CHUNKS FOUND.';
  }

  return chunks
    .map((chunk, idx) => {
      return `--- CONTEXT CHUNK ${idx + 1} [Page ${chunk.pageNumber}, ${chunk.sectionOrPara}] (Relevance Score: ${chunk.score ?? 'N/A'}) ---\n${chunk.text}\n`;
    })
    .join('\n');
}

export function buildUserPrompt(query: string, chunks: DocumentChunk[]): string {
  const context = formatContextChunks(chunks);

  return `USER QUESTION:
"${query}"

DOCUMENT CONTEXT CHUNKS:
${context}

Instructions:
Evaluate the user's question against the context chunks above. Generate the JSON output with:
1. "document_grounded_answer": Factual direct response strictly from chunks (or fallback message if missing).
2. "citations": Array of citation tags corresponding to the chunks used.
3. "supplemental_knowledge_answer": Comprehensive contextual insight and world knowledge.`;
}
