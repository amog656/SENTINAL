from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class InvestigationRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_hops: int = Field(default=3, ge=1)
    results_per_hop: int = Field(default=5, ge=1)


class Clue(BaseModel):
    value: str
    type: str
    source_chunk_id: str
    source_document_id: Optional[str] = None
    hop: int


class EvidenceItem(BaseModel):
    chunk_id: str
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
    text: str = ""
    section: Optional[str] = None
    final_score: float = 0.0
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    retrieved_by: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    bm25_score: Optional[float] = None
    semantic_score: Optional[float] = None
    hop: int = 0
    retrieval_query: str
    trigger_clue: Optional[Clue] = None
    discovered_at_hops: List[int] = Field(default_factory=list)
    discovery_queries: List[str] = Field(default_factory=list)
    discovered_via_clues: List[Clue] = Field(default_factory=list)


class EvidenceRelationship(BaseModel):
    from_chunk_id: str
    to_chunk_id: str
    relationship: str
    clue: Optional[str] = None
    hop: int


class InvestigationHop(BaseModel):
    hop: int
    query: str
    trigger_clue: Optional[Clue] = None
    results_found: int = 0
    new_evidence_found: int = 0


class TraceEvent(BaseModel):
    event: str
    hop: int
    details: Dict[str, Any] = Field(default_factory=dict)


class InvestigationResult(BaseModel):
    query: str
    max_hops: int
    hops_completed: int
    evidence: List[EvidenceItem]
    clues: List[Clue]
    relationships: List[EvidenceRelationship]
    hops: List[InvestigationHop]
    trace: List[TraceEvent]
    summary: Dict[str, int]
