import pdf from 'pdf-parse';
import mammoth from 'mammoth';

export interface ParsedDocumentResult {
  pages: { pageNumber: number; text: string; sectionName?: string }[];
  pageCount: number;
  totalCharacters: number;
  isScannedWarning: boolean;
}

/**
 * Normalizes text extracted from PDF, DOCX, or TXT
 * Cleans excessive whitespace, fixes broken line hyphenations, and standardizes newlines.
 */
function cleanExtractedText(raw: string): string {
  if (!raw) return '';
  return raw
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n')
    .replace(/(\w+)-\n(\w+)/g, '$1$2') // Fix hyphenation at line breaks
    .replace(/[ \t]{2,}/g, ' ') // Collapse multiple horizontal spaces
    .replace(/\n{3,}/g, '\n\n') // Collapse excessive newlines
    .trim();
}

export async function parseDocument(
  buffer: Buffer,
  filename: string,
  mimeType?: string
): Promise<ParsedDocumentResult> {
  const extension = filename.split('.').pop()?.toLowerCase();

  if (extension === 'pdf' || mimeType === 'application/pdf') {
    return parsePDF(buffer);
  } else if (
    extension === 'docx' ||
    mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
  ) {
    return parseDOCX(buffer);
  } else if (extension === 'txt' || mimeType === 'text/plain') {
    return parseTXT(buffer);
  } else {
    throw new Error(`Unsupported file type: .${extension}. Please upload PDF, DOCX, or TXT.`);
  }
}

async function parsePDF(buffer: Buffer): Promise<ParsedDocumentResult> {
  const pages: { pageNumber: number; text: string }[] = [];

  const customPageRender = async (pageData: any) => {
    try {
      const renderOptions = {
        normalizeWhitespace: true,
        disableCombineTextItems: false,
      };

      const textContent = await pageData.getTextContent(renderOptions);
      if (!textContent || !Array.isArray(textContent.items)) {
        return '';
      }

      let lastY: number | null = null;
      let lastX: number | null = null;
      let lastWidth: number = 0;
      let pageText = '';

      for (const item of textContent.items) {
        if (!item || typeof item.str !== 'string') continue;
        const str = item.str;
        if (!str && str !== ' ') continue;

        const hasTransform = Array.isArray(item.transform) && item.transform.length >= 6;
        const currentX = hasTransform ? item.transform[4] : null;
        const currentY = hasTransform ? item.transform[5] : null;

        if (currentY !== null && lastY !== null && Math.abs(lastY - currentY) > 5) {
          // New line
          if (!pageText.endsWith('\n')) {
            pageText += '\n';
          }
        } else if (
          currentX !== null &&
          lastX !== null &&
          currentX > lastX + lastWidth + 2 &&
          !pageText.endsWith(' ') &&
          !pageText.endsWith('\n') &&
          !str.startsWith(' ')
        ) {
          // Spacing between words on same line
          pageText += ' ';
        } else if (
          pageText.length > 0 &&
          !pageText.endsWith(' ') &&
          !pageText.endsWith('\n') &&
          !str.startsWith(' ') &&
          !str.startsWith('.') &&
          !str.startsWith(',') &&
          !str.startsWith(':') &&
          !str.startsWith(';') &&
          !str.startsWith(')') &&
          !str.startsWith(']')
        ) {
          // Fallback word separator if items lack spacing
          pageText += ' ';
        }

        pageText += str;

        if (currentX !== null) lastX = currentX;
        if (currentY !== null) lastY = currentY;
        lastWidth = typeof item.width === 'number' ? item.width : str.length * 5;
      }

      const pageNum = pageData.pageNumber || (pages.length + 1);
      const cleaned = cleanExtractedText(pageText);

      if (cleaned.length > 0) {
        pages.push({
          pageNumber: pageNum,
          text: cleaned,
        });
      }

      return cleaned;
    } catch (err) {
      console.warn('Error in custom page render, returning empty for page:', err);
      return '';
    }
  };

  let numPages = 1;

  try {
    const pdfData = await pdf(buffer, {
      pagerender: customPageRender,
      max: 0,
    });
    numPages = pdfData.numpages || pages.length || 1;
  } catch (err) {
    console.warn('Custom PDF parsing failed, trying default pdf-parse:', err);
  }

  // If custom pagerender did not extract sufficient text, fallback to default pdf-parse
  if (pages.length === 0 || pages.reduce((acc, p) => acc + p.text.length, 0) < 50) {
    try {
      const defaultData = await pdf(buffer);
      const defaultText = cleanExtractedText(defaultData.text || '');
      numPages = defaultData.numpages || 1;

      if (defaultText.length > 0) {
        pages.length = 0; // Clear empty pages

        // Split text evenly across pages or by form-feed
        const formFeedParts = defaultText.split(/\f|\x0C/);
        if (formFeedParts.length > 1) {
          formFeedParts.forEach((part, idx) => {
            const cleanPart = cleanExtractedText(part);
            if (cleanPart) {
              pages.push({ pageNumber: idx + 1, text: cleanPart });
            }
          });
        } else {
          // Approximate pages by chunks of ~2000 characters
          const paragraphs = defaultText.split(/\n\s*\n/).filter((p) => p.trim().length > 0);
          let currPage = 1;
          let currBuf: string[] = [];
          let currLen = 0;

          for (const para of paragraphs) {
            if (currLen + para.length > 2500 && currBuf.length > 0) {
              pages.push({ pageNumber: currPage++, text: currBuf.join('\n\n') });
              currBuf = [para];
              currLen = para.length;
            } else {
              currBuf.push(para);
              currLen += para.length;
            }
          }
          if (currBuf.length > 0) {
            pages.push({ pageNumber: currPage, text: currBuf.join('\n\n') });
          }
        }
      }
    } catch (fallbackErr) {
      console.error('Default pdf-parse fallback also failed:', fallbackErr);
    }
  }

  // Sort pages by pageNumber
  pages.sort((a, b) => a.pageNumber - b.pageNumber);

  const totalChars = pages.reduce((acc, p) => acc + p.text.length, 0);
  const isScannedWarning = totalChars < 50;

  return {
    pages: pages.length > 0 ? pages : [{ pageNumber: 1, text: '' }],
    pageCount: numPages,
    totalCharacters: totalChars,
    isScannedWarning,
  };
}

async function parseDOCX(buffer: Buffer): Promise<ParsedDocumentResult> {
  const result = await mammoth.extractRawText({ buffer });
  const rawText = cleanExtractedText(result.value || '');

  const paragraphs = rawText.split(/\n\s*\n/).filter((p) => p.trim().length > 0);
  const pages: { pageNumber: number; text: string; sectionName?: string }[] = [];

  let currentPage = 1;
  let currentBuffer: string[] = [];
  let currentLen = 0;
  const charsPerPage = 2500;

  for (const para of paragraphs) {
    if (currentLen + para.length > charsPerPage && currentBuffer.length > 0) {
      pages.push({
        pageNumber: currentPage++,
        text: currentBuffer.join('\n\n'),
      });
      currentBuffer = [para];
      currentLen = para.length;
    } else {
      currentBuffer.push(para);
      currentLen += para.length;
    }
  }

  if (currentBuffer.length > 0) {
    pages.push({
      pageNumber: currentPage,
      text: currentBuffer.join('\n\n'),
    });
  }

  const totalChars = rawText.length;
  return {
    pages: pages.length > 0 ? pages : [{ pageNumber: 1, text: rawText }],
    pageCount: pages.length || 1,
    totalCharacters: totalChars,
    isScannedWarning: totalChars < 30,
  };
}

async function parseTXT(buffer: Buffer): Promise<ParsedDocumentResult> {
  const rawText = cleanExtractedText(buffer.toString('utf-8'));
  const sections = rawText.split(/\n\s*\n/).filter((s) => s.trim().length > 0);
  const pages: { pageNumber: number; text: string; sectionName?: string }[] = [];

  let currentPage = 1;
  let currentBuffer: string[] = [];
  let currentLen = 0;
  const charsPerPage = 2000;

  for (const section of sections) {
    if (currentLen + section.length > charsPerPage && currentBuffer.length > 0) {
      pages.push({
        pageNumber: currentPage++,
        text: currentBuffer.join('\n\n'),
      });
      currentBuffer = [section];
      currentLen = section.length;
    } else {
      currentBuffer.push(section);
      currentLen += section.length;
    }
  }

  if (currentBuffer.length > 0) {
    pages.push({
      pageNumber: currentPage,
      text: currentBuffer.join('\n\n'),
    });
  }

  const totalChars = rawText.length;
  return {
    pages: pages.length > 0 ? pages : [{ pageNumber: 1, text: rawText }],
    pageCount: pages.length || 1,
    totalCharacters: totalChars,
    isScannedWarning: totalChars < 20,
  };
}
