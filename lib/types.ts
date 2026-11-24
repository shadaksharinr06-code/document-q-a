export interface DocumentMetadata {
  id: string;
  name: string;
  size: number;
  type: string;
  pageCount: number;
  chunkCount: number;
  uploadedAt: string;
  isScannedWarning?: boolean;
  status: 'idle' | 'uploading' | 'extracting' | 'embedding' | 'ready' | 'error';
  errorMessage?: string;
  chunks?: DocumentChunk[];
}

export interface DocumentChunk {
  id: string;
  text: string;
  pageNumber: number;
  sectionOrPara: string;
  sourceFile: string;
  embedding?: number[];
  score?: number;
}

export interface DualRAGResponse {
  document_grounded_answer: string;
  citations: string[];
  supplemental_knowledge_answer: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  response?: DualRAGResponse;
  retrievedChunks?: DocumentChunk[];
  timestamp: string;
  isError?: boolean;
}

export interface UploadProgressEvent {
  step: 'extracting' | 'embedding' | 'ready' | 'error';
  progress: number;
  message: string;
}
