import OpenAI from 'openai';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { DocumentChunk } from './types';

interface SessionData {
  chunks: DocumentChunk[];
  metadata: {
    filename: string;
    pageCount: number;
    chunkCount: number;
    uploadedAt: string;
  };
  bm25Index: {
    avgDocLen: number;
    docLengths: number[];
    docFreqs: Map<string, number>;
    termFreqs: Map<string, number>[];
    totalDocs: number;
  };
  apiKey?: string;
  provider?: 'openai' | 'gemini' | 'local';
}

const sessions = new Map<string, SessionData>();

const STOPWORDS = new Set([
  'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren',
  'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
  'can', 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
  'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself',
  'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most',
  'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our',
  'ours', 'ourselves', 'out', 'over', 'own', 's', 'same', 'she', 'should', 'so', 'some', 'such', 't',
  'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
  'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what',
  'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'you', 'your'
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 1);
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

export class VectorStore {
  /**
   * Builds BM25 search index for all chunks in a session
   */
  private buildBM25Index(chunks: DocumentChunk[]) {
    const totalDocs = chunks.length;
    const docLengths: number[] = [];
    const docFreqs = new Map<string, number>();
    const termFreqs: Map<string, number>[] = [];
    let totalLen = 0;

    for (const chunk of chunks) {
      const tokens = tokenize(chunk.text);
      const tf = new Map<string, number>();
      const seen = new Set<string>();

      for (const t of tokens) {
        tf.set(t, (tf.get(t) || 0) + 1);
        if (!seen.has(t)) {
          seen.add(t);
          docFreqs.set(t, (docFreqs.get(t) || 0) + 1);
        }
      }

      termFreqs.push(tf);
      docLengths.push(tokens.length);
      totalLen += tokens.length;
    }

    const avgDocLen = totalDocs > 0 ? totalLen / totalDocs : 1;

    return {
      avgDocLen,
      docLengths,
      docFreqs,
      termFreqs,
      totalDocs,
    };
  }

  /**
   * BM25 similarity scoring between query and chunks
   */
  private scoreBM25(
    queryTokens: string[],
    rawQuery: string,
    index: SessionData['bm25Index'],
    chunks: DocumentChunk[]
  ): number[] {
    const k1 = 1.2;
    const b = 0.75;
    const N = index.totalDocs;
    const scores = new Array(N).fill(0);

    const cleanRawQuery = rawQuery.toLowerCase().trim();

    for (const token of queryTokens) {
      if (STOPWORDS.has(token) && queryTokens.length > 3) continue;

      const df = index.docFreqs.get(token) || 0;
      if (df === 0) continue;

      // Robertson-Spärck Jones IDF
      const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));

      for (let i = 0; i < N; i++) {
        const tf = index.termFreqs[i].get(token) || 0;
        if (tf === 0) continue;

        const docLen = index.docLengths[i];
        const denom = tf + k1 * (1 - b + b * (docLen / index.avgDocLen));
        scores[i] += idf * ((tf * (k1 + 1)) / denom);
      }
    }

    // Boost exact phrase matches and numerical mentions
    for (let i = 0; i < N; i++) {
      const chunkText = chunks[i].text.toLowerCase();
      if (cleanRawQuery.length > 5 && chunkText.includes(cleanRawQuery)) {
        scores[i] += 5.0; // Major boost for exact phrase containment
      }
    }

    // Normalize BM25 scores to [0, 1] range
    const maxScore = Math.max(...scores, 0.0001);
    return scores.map((s) => s / maxScore);
  }

  public async generateOpenAIEmbeddings(
    texts: string[],
    apiKey?: string
  ): Promise<number[][] | null> {
    const key = apiKey || process.env.OPENAI_API_KEY;
    if (!key) return null;

    try {
      const openai = new OpenAI({ apiKey: key });
      const response = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: texts.map((t) => t.slice(0, 8000).replace(/\n/g, ' ')),
      });
      return response.data.map((d) => d.embedding);
    } catch (err) {
      console.warn('OpenAI embedding generation failed:', err);
      return null;
    }
  }

  public async storeDocumentChunks(
    sessionId: string,
    chunks: DocumentChunk[],
    metadata: { filename: string; pageCount: number; chunkCount: number },
    options: { apiKey?: string; provider?: 'openai' | 'gemini' | 'local' } = {}
  ): Promise<void> {
    const bm25Index = this.buildBM25Index(chunks);

    // Generate dense embeddings if API key is provided
    let embeddings: number[][] | null = null;
    if (options.apiKey || process.env.OPENAI_API_KEY) {
      const texts = chunks.map((c) => c.text);
      embeddings = await this.generateOpenAIEmbeddings(texts, options.apiKey);
    }

    const enrichedChunks: DocumentChunk[] = chunks.map((c, i) => ({
      ...c,
      embedding: embeddings ? embeddings[i] : undefined,
    }));

    sessions.set(sessionId, {
      chunks: enrichedChunks,
      metadata: {
        ...metadata,
        uploadedAt: new Date().toISOString(),
      },
      bm25Index,
      apiKey: options.apiKey,
      provider: options.provider || (options.apiKey ? 'openai' : 'local'),
    });
  }

  public async similaritySearch(
    sessionId: string,
    query: string,
    topK = 6,
    queryApiKey?: string
  ): Promise<DocumentChunk[]> {
    const session = sessions.get(sessionId);
    if (!session || session.chunks.length === 0) {
      return [];
    }

    const queryTokens = tokenize(query);
    const bm25Scores = this.scoreBM25(queryTokens, query, session.bm25Index, session.chunks);

    // Compute dense similarity if embeddings are active
    const activeKey = queryApiKey || session.apiKey || process.env.OPENAI_API_KEY;
    let queryEmbedding: number[] | null = null;

    if (activeKey && session.chunks[0]?.embedding) {
      try {
        const openai = new OpenAI({ apiKey: activeKey });
        const res = await openai.embeddings.create({
          model: 'text-embedding-3-small',
          input: query.replace(/\n/g, ' '),
        });
        queryEmbedding = res.data[0]?.embedding || null;
      } catch (err) {
        console.warn('Could not generate query embedding, using BM25:', err);
      }
    }

    // Hybrid Fusion
    const scoredChunks = session.chunks.map((chunk, idx) => {
      let finalScore = bm25Scores[idx];

      if (queryEmbedding && chunk.embedding) {
        const denseScore = cosineSimilarity(queryEmbedding, chunk.embedding);
        // Hybrid: 60% BM25 keyword precision + 40% dense semantic similarity
        finalScore = 0.6 * bm25Scores[idx] + 0.4 * Math.max(0, denseScore);
      }

      return {
        ...chunk,
        score: Math.round(finalScore * 1000) / 1000,
      };
    });

    // Sort descending
    scoredChunks.sort((a, b) => (b.score || 0) - (a.score || 0));

    return scoredChunks.slice(0, topK);
  }

  public getSession(sessionId: string): SessionData | undefined {
    return sessions.get(sessionId);
  }

  public clearSession(sessionId: string): boolean {
    return sessions.delete(sessionId);
  }
}

export const vectorStore = new VectorStore();
