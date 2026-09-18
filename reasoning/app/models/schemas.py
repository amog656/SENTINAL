from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


DocumentType = Literal[
    "incident_report", "deployment_note", "architecture_doc", "troubleshooting_guide",
    "customer_complaint", "engineering_discussion", "postmortem",
]
AuthorityType = Literal["postmortem", "deployment_review", "troubleshooting", "discussion"]
InvestigationStatus = Literal["in_progress", "evidence_supported", "insufficient_evidence", "unresolved"]
GoalStatus = Literal["supported", "unsupported", "unresolved", "not_investigated"]


class Document(BaseModel):
    """Frozen RETRIEVAL -> REASONING document contract. Do not add fields casually."""

    document_id: str
    document_type: DocumentType
    title: str
    date: date
    timestamp: datetime | None = None
    service: str | None = None
    version: str | None = None
    authority_type: AuthorityType
    supersedes: str | None = None
    claims: list[str]
    content: str


class Evidence(BaseModel):
    document_id: str
    document_type: DocumentType
    facts: dict[str, Any]
    claims: list[str]
    events: list[dict[str, Any]] = Field(default_factory=list)


class InvestigationPlan(BaseModel):
    initial_query: str
    goals: list["InvestigationGoal"]
    max_hops: int


class InvestigationGoal(BaseModel):
    id: str
    description: str
    critical_to_question: bool = True
    status: GoalStatus = "not_investigated"
    evidence_ids: list[str] = Field(default_factory=list)


class SearchRecord(BaseModel):
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    result_document_ids: list[str] = Field(default_factory=list)


class InvestigationEvent(BaseModel):
    investigation_id: str
    step_type: Literal[
        "plan_created", "search_issued", "document_analyzed", "clue_extracted",
        "follow_up_search", "timeline_updated", "historical_comparison",
        "contradiction_found", "conflict_resolved", "goal_status_changed", "confidence_updated",
        "sufficiency_checked", "report_ready",
    ]
    description: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class InvestigationState(BaseModel):
    """The investigation notebook. Events and the final report read this same state."""

    investigation_id: str
    original_question: str
    status: InvestigationStatus = "in_progress"
    investigation_plan: InvestigationPlan | None = None
    goals: list[InvestigationGoal] = Field(default_factory=list)
    search_history: list[SearchRecord] = Field(default_factory=list)
    documents_seen: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    extracted_clues: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)
    versions: list[str] = Field(default_factory=list)
    incidents: list[str] = Field(default_factory=list)
    deployments: list[str] = Field(default_factory=list)
    technical_events: list[dict[str, Any]] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    historical_comparisons: list[dict[str, Any]] = Field(default_factory=list)
    contradictions: list[dict[str, Any]] = Field(default_factory=list)
    resolved_conflicts: list[dict[str, Any]] = Field(default_factory=list)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_factors: dict[str, Any] = Field(default_factory=dict)
    evidence_status: str = "not_checked"
    loop_termination_reason: Literal["goals_covered", "max_hops_reached", "no_targetable_goal"] | None = None
    open_questions: list[str] = Field(default_factory=list)
    investigation_steps: list[InvestigationEvent] = Field(default_factory=list)


class InvestigateRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class InvestigationStarted(BaseModel):
    investigation_id: str


class ContinueInvestigationRequest(BaseModel):
    investigation_id: str
    instruction: str = Field(min_length=3, max_length=2000)


class TimelineRequest(BaseModel):
    documents: list[Document]


class CompareIncidentsRequest(BaseModel):
    current: Document
    historical: Document


class InvestigationReport(BaseModel):
    investigation_id: str
    question: str | None = None
    status: InvestigationStatus
    conclusion: str | None = None
    confidence: float | None = None
    confidence_factors: dict[str, Any] = Field(default_factory=dict)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    key_evidence: list[dict[str, Any]] = Field(default_factory=list)
    historical_comparisons: list[dict[str, Any]] = Field(default_factory=list)
    contradictions: list[dict[str, Any]] = Field(default_factory=list)
    resolved_conflicts: list[dict[str, Any]] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    investigation_steps: list[InvestigationEvent] = Field(default_factory=list)
    source_document_ids: list[str] = Field(default_factory=list)
    goals: list[InvestigationGoal] = Field(default_factory=list)
    loop_termination_reason: str | None = None
    what_is_known: list[str] = Field(default_factory=list)
    what_is_not_established: list[str] = Field(default_factory=list)


class InvestigationStepsResponse(BaseModel):
    investigation_id: str
    status: InvestigationStatus
    steps: list[InvestigationEvent]


class TimelineResponse(BaseModel):
    timeline: list[dict[str, Any]]


class IncidentComparisonResponse(BaseModel):
    current_document_id: str
    historical_document_id: str
    similarity: Literal["high", "medium", "low", "unknown"]
    conclusion: str
    source_document_ids: list[str]
