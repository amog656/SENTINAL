from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.investigation.models import InvestigationResult


Confidence = Literal["low", "medium", "high"]
AnalysisStatus = Literal["completed", "insufficient_evidence", "llm_unavailable", "validation_failed"]
ClaimType = Literal["observed_fact", "documented_event", "inference", "hypothesis"]


class AgentInvestigationRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_hops: int = Field(default=3, ge=1)
    results_per_hop: int = Field(default=5, ge=1)
    include_trace: bool = True

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be empty")
        return value


class EvidenceReference(BaseModel):
    evidence_id: str
    document_id: Optional[str] = None
    chunk_id: str
    hop: int
    section: Optional[str] = None
    score: float = 0.0
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidencePacket(BaseModel):
    text: str
    evidence: List[EvidenceReference]
    relationships: List[Dict[str, Any]] = Field(default_factory=list)

    @property
    def valid_ids(self) -> set[str]:
        return {item.evidence_id for item in self.evidence}


class Finding(BaseModel):
    claim: str
    type: ClaimType
    evidence_ids: List[str] = Field(default_factory=list)

    @field_validator("claim")
    @classmethod
    def claim_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claim cannot be empty")
        return value


class TimelineEvent(BaseModel):
    date: Optional[str] = None
    event: str
    evidence_ids: List[str] = Field(default_factory=list)

    @field_validator("event")
    @classmethod
    def event_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("event cannot be empty")
        return value


class CausalStep(BaseModel):
    step: str
    type: ClaimType
    evidence_ids: List[str] = Field(default_factory=list)


class Contradiction(BaseModel):
    type: str
    description: str
    evidence_ids: List[str] = Field(default_factory=list)
    claims: List[str] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class ReasoningAnalysis(BaseModel):
    status: AnalysisStatus
    summary: str = ""
    confidence: Optional[Confidence] = None
    evidence_coverage: float = 0.0
    findings: List[Finding] = Field(default_factory=list)
    timeline: List[TimelineEvent] = Field(default_factory=list)
    causal_chain: List[CausalStep] = Field(default_factory=list)
    contradictions: List[Contradiction] = Field(default_factory=list)
    hypotheses: List[Finding] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    conclusion: str = ""
    evidence_used: List[str] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    status: AnalysisStatus
    query: str
    analysis: ReasoningAnalysis
    investigation: InvestigationResult
    evidence_packet: EvidencePacket
    message: Optional[str] = None
