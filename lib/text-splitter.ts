import { DocumentChunk } from './types';

export interface SplitOptions {
  chunkSize?: number;
  chunkOverlap?: number;
}

export class RecursiveTextSplitter {
  private chunkSize: number;
  private chunkOverlap: number;
  private separators: string[];

  constructor(options: SplitOptions = {}) {
    this.chunkSize = options.chunkSize ?? 1000;
    this.chunkOverlap = options.chunkOverlap ?? 150;
    this.separators = ['\n\n', '\n', '. ', ' ', ''];
  }

  public splitText(text: string): string[] {
    return this.split(text, this.separators);
  }

  private split(text: string, separators: string[]): string[] {
    const finalChunks: string[] = [];
    let separator = separators[separators.length - 1];
    let newSeparators: string[] = [];

    for (let i = 0; i < separators.length; i++) {
      const s = separators[i];
      if (s === '') {
        separator = s;
        break;
      }
      if (text.includes(s)) {
        separator = s;
        newSeparators = separators.slice(i + 1);
        break;
      }
    }

    const splits = separator ? text.split(separator) : Array.from(text);
    let currentDoc: string[] = [];
    let currentLen = 0;

    for (const piece of splits) {
      const pieceLen = piece.length;
      if (currentLen + pieceLen + (currentDoc.length > 0 ? separator.length : 0) <= this.chunkSize) {
        currentDoc.push(piece);
        currentLen += pieceLen + (currentDoc.length > 1 ? separator.length : 0);
      } else {
        if (currentDoc.length > 0) {
          const docStr = currentDoc.join(separator).trim();
          if (docStr.length > 0) {
            finalChunks.push(docStr);
          }
          // Handle overlap
          while (currentLen > this.chunkOverlap && currentDoc.length > 0) {
            const popped = currentDoc.shift();
            if (popped) {
              currentLen -= popped.length + (currentDoc.length > 0 ? separator.length : 0);
            }
          }
        }

        if (pieceLen > this.chunkSize && newSeparators.length > 0) {
          const subChunks = this.split(piece, newSeparators);
          finalChunks.push(...subChunks);
        } else {
          currentDoc.push(piece);
          currentLen += pieceLen;
        }
      }
    }

    if (currentDoc.length > 0) {
      const remaining = currentDoc.join(separator).trim();
      if (remaining.length > 0) {
        finalChunks.push(remaining);
      }
    }

    return finalChunks;
  }

  public createChunksFromPages(
    pages: { pageNumber: number; text: string; sectionName?: string }[],
    filename: string
  ): DocumentChunk[] {
    const chunks: DocumentChunk[] = [];
    let chunkCounter = 1;

    for (const page of pages) {
      const rawText = page.text.trim();
      if (!rawText) continue;

      const paragraphs = rawText.split(/\n\s*\n/);
      let paraCounter = 1;

      for (const para of paragraphs) {
        const trimmedPara = para.trim();
        if (!trimmedPara) continue;

        if (trimmedPara.length <= this.chunkSize) {
          chunks.push({
            id: `chk_${chunkCounter++}`,
            text: trimmedPara,
            pageNumber: page.pageNumber,
            sectionOrPara: page.sectionName || `Para ${paraCounter++}`,
            sourceFile: filename,
          });
        } else {
          const subPieces = this.splitText(trimmedPara);
          for (let sIdx = 0; sIdx < subPieces.length; sIdx++) {
            chunks.push({
              id: `chk_${chunkCounter++}`,
              text: subPieces[sIdx],
              pageNumber: page.pageNumber,
              sectionOrPara: `${page.sectionName || `Para ${paraCounter}`} (Part ${sIdx + 1})`,
              sourceFile: filename,
            });
          }
          paraCounter++;
        }
      }
    }

    return chunks;
  }
}
