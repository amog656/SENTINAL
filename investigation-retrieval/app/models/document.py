from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import hashlib


class DocumentData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "PM-211",
                "filename": "Postmortem_May_2026.pdf",
                "document_type": "pdf",
                "text": "PAGE 1\nIncident postmortem for May 2026...",
                "source": "upload",
                "file_size": 48321,
                "page_count": 8,
                "character_count": 18342,
                "metadata": {},
                "created_at": "2026-05-15T10:30:00Z"
            }
        }
    )

    document_id: str = Field(..., description="Deterministic document identifier")
    filename: str = Field(..., description="Original filename")
    document_type: str = Field(..., description="File type: pdf, docx, txt, json")
    text: str = Field(..., description="Extracted and cleaned text content")
    source: str = Field(default="upload", description="Source of the document")
    file_size: int = Field(..., description="File size in bytes")
    page_count: Optional[int] = Field(default=None, description="Number of pages (for PDF)")
    character_count: int = Field(..., description="Number of characters in extracted text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Ingestion timestamp")


class ExtractedMetadata(BaseModel):
    services: List[str] = Field(default_factory=list)
    versions: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    document_type: str = "unknown"
    incident_ids: List[str] = Field(default_factory=list)
    deployment_ids: List[str] = Field(default_factory=list)
    technical_entities: List[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    section: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def generate_document_id(content: bytes, filename: str, document_type: str, existing_id: Optional[str] = None) -> str:
    if existing_id:
        return existing_id
    
    content_hash = hashlib.sha256(content).hexdigest()[:12]
    name_part = filename.rsplit('.', 1)[0] if '.' in filename else filename
    return f"{name_part}-{content_hash}".upper()
