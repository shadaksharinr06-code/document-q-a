import { DocumentChunk, SupplementalSource } from './types';

export interface OnlineSnippet {
  title: string;
  extract: string;
  url: string;
  provider: 'Wikipedia' | 'DuckDuckGo';
}

export interface OnlineKnowledgeResult {
  usedOnline: boolean;
  searchQuery: string;
  snippets: OnlineSnippet[];
  sources: SupplementalSource[];
  formattedForPrompt: string;
  synthesizedAnswer: string;
}

const REQUEST_TIMEOUT_MS = 4500;

async function fetchJsonWithTimeout(url: string, timeoutMs = REQUEST_TIMEOUT_MS): Promise<any | null> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      headers: {
        'User-Agent': 'DocuDualRAG/1.0 (https://document-q-a-t5zz.vercel.app)',
        Accept: 'application/json',
      },
      signal: controller.signal,
    });
    if (!res.ok) return null;
    return await res.json();
  } finally {
    clearTimeout(timeoutId);
  }
}

const QUESTION_STOPWORDS = new Set([
  'a', 'about', 'above', 'according', 'after', 'again', 'against', 'all', 'also', 'am', 'an',
  'and', 'any', 'are', 'as', 'ask', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
  'between', 'both', 'briefly', 'but', 'by', 'can', 'compare', 'could', 'define', 'describe',
  'detail', 'details', 'did', 'do', 'doc', 'document', 'documents', 'does', 'doing', 'down',
  'during', 'each', 'elaborate', 'else', 'etc', 'even', 'explain', 'explanation', 'extra',
  'few', 'file', 'find', 'for', 'from', 'further', 'give', 'had', 'has', 'have', 'having',
  'help', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
  'info', 'information', 'into', 'is', 'it', 'its', 'itself', 'just', 'know', 'list', 'many',
  'may', 'me', 'mean', 'meaning', 'mentioned', 'more', 'most', 'much', 'must', 'my', 'myself',
  'need', 'no', 'nor', 'not', 'note', 'notable', 'now', 'of', 'off', 'on', 'once', 'online',
  'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'overview', 'own', 'page',
  'paper', 'pdf', 'please', 'provide', 'question', 'regarding', 'related', 'same', 'say',
  'says', 'section', 'she', 'should', 'show', 'so', 'some', 'such', 'summarize', 'summary',
  'tell', 'text', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
  'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
  'uploaded', 'use', 'used', 'using', 'very', 'want', 'was', 'we', 'were', 'what', 'whatever',
  'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'work', 'would',
  'you', 'your', 'yours', 'yourself',
]);

const GENERIC_DOC_TERMS = new Set([
  'objective', 'objectives', 'methodology', 'methodologies', 'requirement', 'requirements',
  'guideline', 'guidelines', 'risk', 'risks', 'limitation', 'limitations', 'future', 'step',
  'steps', 'key', 'core', 'main', 'point', 'points', 'conclusion', 'conclusions', 'finding',
  'findings', 'purpose', 'goal', 'goals', 'result', 'results', 'feature', 'features',
  'benefit', 'benefits', 'challenge', 'challenges', 'topic', 'topics', 'concept', 'concepts',
]);

function cleanWhitespace(text: string): string {
  return text.replace(/\s+/g, ' ').trim();
}

function truncateAtSentence(text: string, maxChars: number): string {
  const cleaned = cleanWhitespace(text);
  if (cleaned.length <= maxChars) return cleaned;
  const slice = cleaned.slice(0, maxChars);
  const lastSentenceEnd = Math.max(
    slice.lastIndexOf('. '),
    slice.lastIndexOf('! '),
    slice.lastIndexOf('? ')
  );
  if (lastSentenceEnd > maxChars * 0.45) {
    return slice.slice(0, lastSentenceEnd + 1).trim();
  }
  return `${slice.trim()}...`;
}

/**
 * Extract salient domain keywords from retrieved chunks when the user asks a generic
 * document-level question (e.g., "What is the core objective of this document?").
 */
function extractDocumentTopicTerms(chunks: DocumentChunk[], filename?: string): string[] {
  const termCounts = new Map<string, number>();

  if (filename) {
    const cleanName = filename.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ');
    for (const w of cleanName.split(/\s+/)) {
      const lower = w.toLowerCase().replace(/[^\w]/g, '');
      if (lower.length > 2 && !QUESTION_STOPWORDS.has(lower) && !/^\d+$/.test(lower)) {
        termCounts.set(w, (termCounts.get(w) || 0) + 4);
      }
    }
  }

  for (const chunk of chunks.slice(0, 4)) {
    // Prioritize multi-word capitalized phrases and domain terms
    const phrases = chunk.text.match(/\b[A-Z][a-zA-Z0-9-]{2,}(?:\s+[A-Z][a-zA-Z0-9-]{2,}){0,2}\b/g) || [];
    for (const phrase of phrases) {
      const lower = phrase.toLowerCase();
      if (!QUESTION_STOPWORDS.has(lower) && !GENERIC_DOC_TERMS.has(lower)) {
        termCounts.set(phrase, (termCounts.get(phrase) || 0) + 2);
      }
    }

    const words = chunk.text.toLowerCase().match(/\b[a-z][a-z0-9-]{3,}\b/g) || [];
    for (const word of words) {
      if (!QUESTION_STOPWORDS.has(word) && !GENERIC_DOC_TERMS.has(word)) {
        termCounts.set(word, (termCounts.get(word) || 0) + 1);
      }
    }
  }

  return Array.from(termCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([term]) => term);
}

/**
 * Builds targeted search queries tailored to the user's question (and document topic if generic).
 */
export function buildSearchQueries(
  question: string,
  chunks: DocumentChunk[] = [],
  filename?: string
): { primary: string; fallback: string | null } {
  const rawTokens = question
    .replace(/[?!.,"'()[\]{}:;]/g, ' ')
    .split(/\s+/)
    .map((t) => t.trim())
    .filter(Boolean);

  const meaningfulTokens = rawTokens.filter((token) => {
    const lower = token.toLowerCase();
    return lower.length > 1 && !QUESTION_STOPWORDS.has(lower);
  });

  const specificDomainTokens = meaningfulTokens.filter(
    (t) => !GENERIC_DOC_TERMS.has(t.toLowerCase())
  );

  // If the question itself contains specific entities/concepts, search those directly
  if (specificDomainTokens.length >= 1) {
    const primary = meaningfulTokens.slice(0, 8).join(' ');
    const fallback =
      specificDomainTokens.length >= 2 && specificDomainTokens.length < meaningfulTokens.length
        ? specificDomainTokens.slice(0, 5).join(' ')
        : null;
    return { primary, fallback };
  }

  // Otherwise, anchor the generic question to the document's main topic
  const docTopics = extractDocumentTopicTerms(chunks, filename);
  const intentTerms = meaningfulTokens.slice(0, 2);
  const combined = [...docTopics.slice(0, 3), ...intentTerms].join(' ').trim();

  return {
    primary: combined || cleanWhitespace(question),
    fallback: docTopics.slice(0, 2).join(' ').trim() || null,
  };
}

async function fetchWikipediaSnippets(searchQuery: string): Promise<OnlineSnippet[]> {
  if (!searchQuery.trim()) return [];

  const searchUrl = new URL('https://en.wikipedia.org/w/api.php');
  searchUrl.searchParams.set('action', 'query');
  searchUrl.searchParams.set('list', 'search');
  searchUrl.searchParams.set('srsearch', searchQuery);
  searchUrl.searchParams.set('srlimit', '3');
  searchUrl.searchParams.set('utf8', '1');
  searchUrl.searchParams.set('format', 'json');

  const searchData = await fetchJsonWithTimeout(searchUrl.toString());
  const hits: Array<{ pageid: number; title: string; snippet?: string }> =
    searchData?.query?.search || [];
  if (hits.length === 0) return [];

  const pageIds = hits.slice(0, 2).map((h) => h.pageid).join('|');
  const extractUrl = new URL('https://en.wikipedia.org/w/api.php');
  extractUrl.searchParams.set('action', 'query');
  extractUrl.searchParams.set('prop', 'extracts|info|pageprops');
  extractUrl.searchParams.set('inprop', 'url');
  extractUrl.searchParams.set('exintro', '1');
  extractUrl.searchParams.set('explaintext', '1');
  extractUrl.searchParams.set('exchars', '1200');
  extractUrl.searchParams.set('pageids', pageIds);
  extractUrl.searchParams.set('format', 'json');

  const extractData = await fetchJsonWithTimeout(extractUrl.toString());
  const pagesObj: Record<string, any> = extractData?.query?.pages || {};

  const snippets: OnlineSnippet[] = [];
  for (const hit of hits.slice(0, 2)) {
    const page = pagesObj[String(hit.pageid)];
    if (!page || page.missing !== undefined) continue;
    // Skip disambiguation pages when they have almost no real content
    const isDisambig = page.pageprops && 'disambiguation' in page.pageprops;
    const rawExtract = cleanWhitespace(page.extract || '');
    if (!rawExtract || (isDisambig && rawExtract.length < 180)) continue;

    snippets.push({
      title: page.title || hit.title,
      extract: truncateAtSentence(rawExtract, 700),
      url:
        page.fullurl ||
        `https://en.wikipedia.org/wiki/${encodeURIComponent((page.title || hit.title).replace(/\s+/g, '_'))}`,
      provider: 'Wikipedia',
    });
  }

  return snippets;
}

async function fetchDuckDuckGoSnippets(searchQuery: string): Promise<OnlineSnippet[]> {
  if (!searchQuery.trim()) return [];

  const ddgUrl = new URL('https://api.duckduckgo.com/');
  ddgUrl.searchParams.set('q', searchQuery);
  ddgUrl.searchParams.set('format', 'json');
  ddgUrl.searchParams.set('no_html', '1');
  ddgUrl.searchParams.set('skip_disambig', '1');

  const data = await fetchJsonWithTimeout(ddgUrl.toString());
  if (!data) return [];
  const snippets: OnlineSnippet[] = [];

  if (data?.AbstractText && cleanWhitespace(data.AbstractText).length > 30) {
    snippets.push({
      title: data.Heading || searchQuery,
      extract: truncateAtSentence(data.AbstractText, 650),
      url: data.AbstractURL || `https://duckduckgo.com/?q=${encodeURIComponent(searchQuery)}`,
      provider: 'DuckDuckGo',
    });
  }

  // Pull up to 2 related topic definitions if no primary abstract or to add extra context
  if (Array.isArray(data?.RelatedTopics)) {
    for (const item of data.RelatedTopics) {
      if (snippets.length >= 2) break;
      if (
        item?.Text &&
        item?.FirstURL &&
        !String(item.FirstURL).includes('/c/') &&
        cleanWhitespace(item.Text).length > 45
      ) {
        snippets.push({
          title: item.Text.split(' - ')[0]?.slice(0, 70) || searchQuery,
          extract: truncateAtSentence(item.Text, 300),
          url: item.FirstURL,
          provider: 'DuckDuckGo',
        });
      }
    }
  }

  return snippets;
}

function deduplicateSnippets(snippets: OnlineSnippet[]): OnlineSnippet[] {
  const seenUrls = new Set<string>();
  const seenTitles = new Set<string>();
  const unique: OnlineSnippet[] = [];

  for (const s of snippets) {
    const normUrl = s.url.toLowerCase();
    const normTitle = s.title.toLowerCase();
    if (seenUrls.has(normUrl) || seenTitles.has(normTitle)) continue;
    seenUrls.add(normUrl);
    seenTitles.add(normTitle);
    unique.push(s);
  }
  return unique.slice(0, 3);
}

/**
 * Fetches live online knowledge tailored to the user's question (and document context when needed).
 * Never throws; returns a graceful fallback structure if online providers fail or time out.
 */
export async function fetchOnlineSupplementalKnowledge(
  question: string,
  chunks: DocumentChunk[] = [],
  filename?: string
): Promise<OnlineKnowledgeResult> {
  const { primary, fallback } = buildSearchQueries(question, chunks, filename);

  let collected: OnlineSnippet[] = [];
  let effectiveQuery = primary;

  try {
    const [wikiRes, ddgRes] = await Promise.allSettled([
      fetchWikipediaSnippets(primary),
      fetchDuckDuckGoSnippets(primary),
    ]);

    if (ddgRes.status === 'fulfilled') collected.push(...ddgRes.value);
    if (wikiRes.status === 'fulfilled') collected.push(...wikiRes.value);

    // Retry once with focused fallback query if primary yielded no results
    if (collected.length === 0 && fallback && fallback !== primary) {
      effectiveQuery = fallback;
      const [wikiFallback, ddgFallback] = await Promise.allSettled([
        fetchWikipediaSnippets(fallback),
        fetchDuckDuckGoSnippets(fallback),
      ]);
      if (ddgFallback.status === 'fulfilled') collected.push(...ddgFallback.value);
      if (wikiFallback.status === 'fulfilled') collected.push(...wikiFallback.value);
    }
  } catch {
    // Graceful fallback handled below
  }

  const snippets = deduplicateSnippets(collected);
  const sources: SupplementalSource[] = snippets.map((s) => ({
    title: s.title,
    url: s.url,
    provider: s.provider,
  }));

  if (snippets.length === 0) {
    const docTopics = extractDocumentTopicTerms(chunks, filename);
    const topicHint = docTopics.length > 0 ? docTopics.slice(0, 3).join(', ') : effectiveQuery;
    return {
      usedOnline: false,
      searchQuery: effectiveQuery,
      snippets: [],
      sources: [],
      formattedForPrompt: 'NO LIVE ONLINE SNIPPETS RETRIEVED.',
      synthesizedAnswer: `Extra Context for "${question}":\n\nNo direct external encyclopedia entry matched "${effectiveQuery}" right now. Based on the surrounding subject terms (${topicHint}), you may want to refine the question with a specific concept, standard, or entity name (or connect an OpenAI/Gemini key in LLM Settings for deep AI synthesis).`,
    };
  }

  const formattedForPrompt = snippets
    .map(
      (s, idx) =>
        `[Online Source ${idx + 1} - ${s.provider}: ${s.title}] (${s.url})\n${s.extract}`
    )
    .join('\n\n');

  const primarySnippet = snippets[0];
  const secondarySnippets = snippets.slice(1);

  const parts: string[] = [
    `Extra Online Information for "${question}":`,
    `${primarySnippet.title} (${primarySnippet.provider}):\n${primarySnippet.extract}`,
  ];

  if (secondarySnippets.length > 0) {
    const relatedLines = secondarySnippets.map(
      (s) => `• ${s.title} (${s.provider}): ${s.extract}`
    );
    parts.push(`Additional Related Context:\n${relatedLines.join('\n\n')}`);
  }

  return {
    usedOnline: true,
    searchQuery: effectiveQuery,
    snippets,
    sources,
    formattedForPrompt,
    synthesizedAnswer: parts.join('\n\n'),
  };
}
