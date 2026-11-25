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
1. Question-Tailored Extra Information: Directly expand on the user's specific question with extra, relevant information outside the document—such as clear definitions, background theory, real-world examples, comparisons, industry standards, or practical implications tied to the exact topic asked.
2. Live Online Context Integration: If LIVE ONLINE REFERENCE CONTEXT is provided, weave its relevant facts into "supplemental_knowledge_answer" alongside your broader knowledge. NEVER copy online context into "document_grounded_answer".
3. Avoid generic filler statements; every sentence should add concrete extra value specific to the question.
4. Keep the tone helpful, professional, and clear.`;

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

export function buildUserPrompt(
  query: string,
  chunks: DocumentChunk[],
  onlineContext?: string
): string {
  const context = formatContextChunks(chunks);
  const onlineSection =
    onlineContext && onlineContext !== 'NO LIVE ONLINE SNIPPETS RETRIEVED.'
      ? `\nLive Online Reference Context (Use ONLY for "supplemental_knowledge_answer", NEVER for "document_grounded_answer"):\n${onlineContext}\n`
      : '';

  return `Context:
${context}
${onlineSection}
Question:
${query}

Instructions:
You are a precise Document Q&A Assistant. Answer the user's question using ONLY the provided document context above for "document_grounded_answer".
- If the answer is not explicitly contained in the document context, reply: "I cannot find this information in the provided documents."
- Do not make assumptions or bring in outside/online knowledge for "document_grounded_answer".
- Cite the document name, page, or section if available in "citations".
- For "supplemental_knowledge_answer", provide rich, concrete extra information specifically tailored to the user's question (using your broader knowledge and any Live Online Reference Context above).
Return your response as a valid JSON object matching the requested schema.`;
}
