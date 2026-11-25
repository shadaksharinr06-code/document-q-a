import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { vectorStore } from '@/lib/vector-store';
import { SYSTEM_PROMPT, buildUserPrompt } from '@/lib/prompt-templates';
import {
  fetchOnlineSupplementalKnowledge,
  OnlineKnowledgeResult,
} from '@/lib/online-knowledge';
import { DualRAGResponse, DocumentChunk } from '@/lib/types';

/**
 * Intelligent Extractive QA + Live Online Knowledge Engine used when no external LLM API key is provided.
 * Synthesizes grounded facts from retrieved chunks without hallucination, and pairs them with
 * live online knowledge tailored specifically to the user's question.
 */
function synthesizeOfflineAnswer(
  query: string,
  chunks: DocumentChunk[],
  filename: string,
  onlineResult: OnlineKnowledgeResult
): DualRAGResponse {
  const supplementalFields = {
    supplemental_knowledge_answer: onlineResult.synthesizedAnswer,
    supplemental_sources: onlineResult.sources,
    supplemental_mode: (onlineResult.usedOnline ? 'online' : 'offline') as 'online' | 'offline',
  };

  if (chunks.length === 0) {
    return {
      document_grounded_answer: 'I cannot find this information in the provided documents.',
      citations: [],
      ...supplementalFields,
    };
  }

  const queryTerms = query
    .toLowerCase()
    .replace(/[^\w\s]/g, '')
    .split(/\s+/)
    .filter((w) => w.length > 2);

  // Score sentences within retrieved chunks
  const candidateSentences: { text: string; page: number; para: string; score: number }[] = [];

  for (const chunk of chunks) {
    const rawSentences = chunk.text
      .split(/(?<=[.?!])\s+(?=[A-Z0-9])|\n+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 15);

    for (const sent of rawSentences) {
      const sentLower = sent.toLowerCase();
      let matchCount = 0;

      for (const term of queryTerms) {
        if (sentLower.includes(term)) {
          matchCount += 1;
        }
      }

      if (matchCount > 0) {
        const density = matchCount / Math.max(1, queryTerms.length);
        candidateSentences.push({
          text: sent,
          page: chunk.pageNumber,
          para: chunk.sectionOrPara,
          score: (chunk.score || 0) * 0.5 + density * 0.5,
        });
      }
    }
  }

  candidateSentences.sort((a, b) => b.score - a.score);

  const topSentences = candidateSentences.slice(0, 4);

  if (topSentences.length === 0 || (topSentences[0]?.score || 0) < 0.1) {
    // If top chunk has high score, present its relevant core
    if (chunks[0] && (chunks[0].score || 0) > 0.3) {
      const bestChunk = chunks[0];
      return {
        document_grounded_answer: `According to ${filename} (Page ${bestChunk.pageNumber}, ${bestChunk.sectionOrPara}):\n\n"${bestChunk.text}"`,
        citations: [`Page ${bestChunk.pageNumber}, ${bestChunk.sectionOrPara}`],
        ...supplementalFields,
      };
    }

    return {
      document_grounded_answer: 'I cannot find this information in the provided documents.',
      citations: [],
      ...supplementalFields,
    };
  }

  // Deduplicate and assemble grounded answer
  const seenTexts = new Set<string>();
  const citationsSet = new Set<string>();
  const assembledPoints: string[] = [];

  for (const s of topSentences) {
    if (!seenTexts.has(s.text)) {
      seenTexts.add(s.text);
      citationsSet.add(`Page ${s.page}, ${s.para}`);
      assembledPoints.push(`• "${s.text}" [Page ${s.page}, ${s.para}]`);
    }
  }

  const citations = Array.from(citationsSet);
  const groundedAnswer = `Direct findings extracted from ${filename}:\n\n${assembledPoints.join('\n\n')}`;

  return {
    document_grounded_answer: groundedAnswer,
    citations,
    ...supplementalFields,
  };
}

async function generateWithGemini(apiKey: string, userPrompt: string): Promise<string> {
  const genAI = new GoogleGenerativeAI(apiKey);
  const candidateModels = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash'];
  let lastError: unknown;

  for (const modelName of candidateModels) {
    try {
      const model = genAI.getGenerativeModel({
        model: modelName,
        generationConfig: {
          responseMimeType: 'application/json',
          temperature: 0.1,
        },
        systemInstruction: SYSTEM_PROMPT,
      });
      const result = await model.generateContent(userPrompt);
      return result.response.text();
    } catch (err: any) {
      lastError = err;
      const msg = String(err?.message || '');
      const isModelNotFound = msg.includes('404') || msg.toLowerCase().includes('not found');
      if (!isModelNotFound) {
        throw err;
      }
    }
  }

  throw lastError;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { query, sessionId, apiKey, provider = 'openai' } = body;

    // Check for Vercel internal service binding: BACKEND_URL
    const backendUrl = process.env.BACKEND_URL;
    if (backendUrl) {
      try {
        const targetUrl = new URL('/api/query', backendUrl);
        const backendRes = await fetch(targetUrl.toString(), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (backendRes.ok) {
          const data = await backendRes.json();
          return NextResponse.json(data);
        }
      } catch (proxyErr) {
        console.warn('Proxy to BACKEND_URL failed, falling back to local handler:', proxyErr);
      }
    }

    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      return NextResponse.json({ error: 'Query cannot be empty.' }, { status: 400 });
    }

    if (!sessionId) {
      return NextResponse.json(
        { error: 'No active document session. Please upload a document first.' },
        { status: 400 }
      );
    }

    let session = vectorStore.getSession(sessionId);

    // If serverless container cold-started or recycled, rehydrate seamlessly from client chunks
    if ((!session || session.chunks.length === 0) && body.chunks && Array.isArray(body.chunks) && body.chunks.length > 0) {
      await vectorStore.storeDocumentChunks(
        sessionId,
        body.chunks,
        {
          filename: body.filename || 'Uploaded Document',
          pageCount: body.chunks[body.chunks.length - 1]?.pageNumber || 1,
          chunkCount: body.chunks.length,
        },
        { apiKey, provider }
      );
      session = vectorStore.getSession(sessionId);
    }

    if (!session || session.chunks.length === 0) {
      return NextResponse.json(
        { error: 'Session expired or document not found. Please re-upload.' },
        { status: 404 }
      );
    }

    const cleanQuery = query.trim();

    // 1. Retrieve top-6 relevant chunks using Hybrid BM25 + Vector Search
    const retrievedChunks = await vectorStore.similaritySearch(
      sessionId,
      cleanQuery,
      6,
      apiKey
    );

    // 2. Fetch live online supplemental knowledge tailored to the user's question
    const onlineResult = await fetchOnlineSupplementalKnowledge(
      cleanQuery,
      retrievedChunks,
      session.metadata.filename
    );

    const activeApiKey = apiKey || session.apiKey || process.env.OPENAI_API_KEY || process.env.GEMINI_API_KEY;
    const isGemini = provider === 'gemini' || (activeApiKey && activeApiKey.startsWith('AIza'));

    const buildLlmDualResponse = (rawJsonText: string): DualRAGResponse => {
      const parsed = JSON.parse(rawJsonText || '{}');
      return {
        document_grounded_answer:
          parsed.document_grounded_answer ||
          'I cannot find this information in the provided documents.',
        citations: Array.isArray(parsed.citations) ? parsed.citations : [],
        supplemental_knowledge_answer:
          parsed.supplemental_knowledge_answer || onlineResult.synthesizedAnswer,
        supplemental_sources: onlineResult.sources,
        supplemental_mode: onlineResult.usedOnline ? 'llm+online' : 'llm',
      };
    };

    let dualResponse: DualRAGResponse;

    if (activeApiKey) {
      const userPrompt = buildUserPrompt(
        cleanQuery,
        retrievedChunks,
        onlineResult.formattedForPrompt
      );

      if (isGemini) {
        // --- GOOGLE GEMINI EXECUTION ---
        try {
          const rawText = await generateWithGemini(activeApiKey, userPrompt);
          dualResponse = buildLlmDualResponse(rawText);
        } catch (geminiErr: any) {
          console.warn('Gemini inference failed, falling back to online/extractive synthesizer:', geminiErr);
          dualResponse = synthesizeOfflineAnswer(
            cleanQuery,
            retrievedChunks,
            session.metadata.filename,
            onlineResult
          );
        }
      } else {
        // --- OPENAI GPT-4O EXECUTION ---
        try {
          const openai = new OpenAI({ apiKey: activeApiKey });
          const completion = await openai.chat.completions.create({
            model: 'gpt-4o',
            messages: [
              { role: 'system', content: SYSTEM_PROMPT },
              { role: 'user', content: userPrompt },
            ],
            response_format: { type: 'json_object' },
            temperature: 0.1,
          });

          const rawContent = completion.choices[0]?.message?.content || '{}';
          dualResponse = buildLlmDualResponse(rawContent);
        } catch (openAiErr: any) {
          console.warn('OpenAI inference failed, falling back to online/extractive synthesizer:', openAiErr);
          dualResponse = synthesizeOfflineAnswer(
            cleanQuery,
            retrievedChunks,
            session.metadata.filename,
            onlineResult
          );
        }
      }
    } else {
      // --- BUILT-IN EXTRACTIVE QA + LIVE ONLINE KNOWLEDGE ---
      dualResponse = synthesizeOfflineAnswer(
        cleanQuery,
        retrievedChunks,
        session.metadata.filename,
        onlineResult
      );
    }

    return NextResponse.json({
      success: true,
      response: dualResponse,
      retrievedChunks,
    });
  } catch (error: any) {
    console.error('Error in /api/query:', error);
    return NextResponse.json(
      { error: error?.message || 'Failed to process query' },
      { status: 500 }
    );
  }
}
