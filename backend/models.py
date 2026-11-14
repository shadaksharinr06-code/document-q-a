from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

class DualRAGResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    document_grounded_answer: str = Field(
        ...,
        description="Factual response strictly extracted from the provided document chunks."
    )
    citations: List[str] = Field(
        default_factory=list,
        description="List of citation tags indicating specific source pages or paragraphs."
    )
    supplemental_knowledge_answer: str = Field(
        ...,
        description="Broader contextual insights, industry best practices, explanations, or relevant background."
    )

class DocumentMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    name: str
    size: int
    type: str
    page_count: int = Field(..., alias="pageCount")
    chunk_count: int = Field(..., alias="chunkCount")
    uploaded_at: str = Field(..., alias="uploadedAt")
    is_scanned_warning: bool = Field(False, alias="isScannedWarning")
    status: str = "ready"

class ChunkItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    text: str
    page_number: int = Field(..., alias="pageNumber")
    section_or_para: str = Field(..., alias="sectionOrPara")
    source_file: str = Field(..., alias="sourceFile")
    score: Optional[float] = None

class QueryRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    query: str
    session_id: Optional[str] = Field(None, alias="sessionId")
    api_key: Optional[str] = Field(None, alias="apiKey")
    provider: Optional[str] = None
    top_k: int = Field(5, alias="topK")

class QueryResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    success: bool
    response: DualRAGResponse
    retrieved_chunks: List[ChunkItem] = Field(default_factory=list, alias="retrievedChunks")

class ResetRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    session_id: Optional[str] = Field(None, alias="sessionId")
