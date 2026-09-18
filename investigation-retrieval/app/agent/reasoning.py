import json
import re
from datetime import date
from typing import Any

from pydantic import ValidationError

from app.agent.models import (
    AgentResult,
    Contradiction,
    EvidencePacket,
    EvidenceReference,
    ReasoningAnalysis,
)
from app.investigation.models import EvidenceRelationship, InvestigationResult


class ReasoningValidationError(Exception):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def build_evidence_packet(investigation: InvestigationResult, max_evidence: int) -> EvidencePacket:
    selected = _select_evidence(investigation, max_evidence)
    id_by_chunk = {item.chunk_id: f"E{index + 1}" for index, item in enumerate(selected)}
    references = [
        EvidenceReference(
            evidence_id=id_by_chunk[item.chunk_id],
            document_id=item.document_id,
            chunk_id=item.chunk_id,
            hop=item.hop,
            section=item.section,
            score=item.final_score,
            text=item.text,
            metadata=item.metadata,
        )
        for item in selected
    ]
    relationships = [
        _relationship_packet(relationship, id_by_chunk)
        for relationship in investigation.relationships
        if relationship.from_chunk_id in id_by_chunk and relationship.to_chunk_id in id_by_chunk
    ]
    relationships = [relationship for relationship in relationships if relationship]
    return EvidencePacket(text=_format_packet(references, relationships), evidence=references, relationships=relationships)


def parse_reasoning_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReasoningValidationError([f"LLM response was not valid JSON: {exc.msg}"]) from exc
    if not isinstance(parsed, dict):
        raise ReasoningValidationError(["LLM response must be a JSON object"])
    return parsed


def validate_reasoning_output(raw: str, evidence_packet: EvidencePacket) -> ReasoningAnalysis:
    parsed = parse_reasoning_json(raw)
    try:
        analysis = ReasoningAnalysis.model_validate(parsed)
    except ValidationError as exc:
        raise ReasoningValidationError([str(error["msg"]) for error in exc.errors()]) from exc

    deterministic_contradictions = detect_contradictions(evidence_packet)
    analysis.contradictions = _merge_contradictions(analysis.contradictions, deterministic_contradictions)
    errors = _validate_evidence_references(analysis, evidence_packet.valid_ids)
    if errors:
        analysis.status = "validation_failed"
        analysis.validation_errors = errors
        raise ReasoningValidationError(errors)

    analysis.timeline = sorted(analysis.timeline, key=lambda event: (event.date or "9999-99-99", event.event))
    analysis.evidence_used = sorted(set(analysis.evidence_used))
    analysis.evidence_coverage = calculate_evidence_coverage(analysis)
    return analysis


def validation_failed_analysis(errors: list[str]) -> ReasoningAnalysis:
    return ReasoningAnalysis(
        status="validation_failed",
        summary="The reasoning response failed validation.",
        confidence="low",
        validation_errors=errors,
        conclusion="No evidence-backed conclusion was accepted because validation failed.",
    )


def unavailable_analysis() -> ReasoningAnalysis:
    return ReasoningAnalysis(
        status="llm_unavailable",
        summary="Investigation retrieval completed, but no reasoning provider is configured.",
        confidence="low",
        conclusion="Reasoning was not performed because the LLM provider is unavailable.",
    )


def calculate_evidence_coverage(analysis: ReasoningAnalysis) -> float:
    claims = [*analysis.findings, *analysis.timeline, *analysis.causal_chain, *analysis.hypotheses]
    if not claims:
        return 0.0
    supported = sum(1 for claim in claims if getattr(claim, "evidence_ids", []))
    return round(supported / len(claims), 4)


def detect_contradictions(evidence_packet: EvidencePacket) -> list[Contradiction]:
    contradictions = []
    contradictions.extend(_detect_deployment_time_conflicts(evidence_packet))
    contradictions.extend(_detect_database_saturation_conflicts(evidence_packet))
    return contradictions


def _select_evidence(investigation: InvestigationResult, max_evidence: int):
    connected_chunks = {relationship.from_chunk_id for relationship in investigation.relationships}
    connected_chunks.update(relationship.to_chunk_id for relationship in investigation.relationships)
    selected = []
    seen_documents = set()
    sorted_evidence = sorted(
        investigation.evidence,
        key=lambda item: (
            -item.final_score,
            0 if item.chunk_id in connected_chunks else 1,
            len(item.discovered_at_hops),
            item.chunk_id,
        ),
    )
    for item in sorted_evidence:
        if len(selected) >= max_evidence:
            break
        selected.append(item)
        if item.document_id:
            seen_documents.add(item.document_id)

    if len(selected) >= max_evidence:
        return sorted(selected, key=lambda item: (-item.final_score, item.chunk_id))

    for item in sorted(investigation.evidence, key=lambda item: (item.document_id or "", -item.final_score, item.chunk_id)):
        if item in selected or item.document_id in seen_documents:
            continue
        selected.append(item)
        if len(selected) >= max_evidence:
            break
    return sorted(selected, key=lambda item: (-item.final_score, item.chunk_id))


def _format_packet(references: list[EvidenceReference], relationships: list[dict[str, Any]]) -> str:
    blocks = []
    for reference in references:
        metadata = reference.metadata or {}
        blocks.append(
            "\n".join(
                [
                    f"EVIDENCE [{reference.evidence_id}]",
                    f"document_id: {reference.document_id}",
                    f"chunk_id: {reference.chunk_id}",
                    f"hop: {reference.hop}",
                    f"section: {reference.section}",
                    f"score: {reference.score}",
                    "",
                    "TEXT:",
                    reference.text,
                    "",
                    "METADATA:",
                    json.dumps(metadata, sort_keys=True),
                ]
            )
        )
    if relationships:
        blocks.append("RELATIONSHIPS:\n" + json.dumps(relationships, sort_keys=True))
    return "\n\n".join(blocks)


def _relationship_packet(relationship: EvidenceRelationship, id_by_chunk: dict[str, str]) -> dict[str, Any]:
    return {
        "from": id_by_chunk[relationship.from_chunk_id],
        "to": id_by_chunk[relationship.to_chunk_id],
        "relationship": relationship.relationship,
        "clue": relationship.clue,
        "hop": relationship.hop,
    }


def _validate_evidence_references(analysis: ReasoningAnalysis, valid_ids: set[str]) -> list[str]:
    errors = []
    referenced = []
    items = [
        *[(finding, "finding") for finding in analysis.findings],
        *[(event, "timeline") for event in analysis.timeline],
        *[(step, "causal_chain") for step in analysis.causal_chain],
        *[(hypothesis, "hypothesis") for hypothesis in analysis.hypotheses],
        *[(contradiction, "contradiction") for contradiction in analysis.contradictions],
    ]
    for item, label in items:
        evidence_ids = getattr(item, "evidence_ids", [])
        if label in {"finding", "timeline", "causal_chain"} and not evidence_ids:
            errors.append(f"{label} item must cite evidence")
        for evidence_id in evidence_ids:
            if evidence_id not in valid_ids:
                errors.append(f"Unknown evidence id referenced: {evidence_id}")
            else:
                referenced.append(evidence_id)
    for evidence_id in analysis.evidence_used:
        if evidence_id not in valid_ids:
            errors.append(f"Unknown evidence id referenced: {evidence_id}")
        else:
            referenced.append(evidence_id)
    if not analysis.evidence_used:
        analysis.evidence_used = sorted(set(referenced))
    return sorted(set(errors))


def _merge_contradictions(first: list[Contradiction], second: list[Contradiction]) -> list[Contradiction]:
    seen = set()
    merged = []
    for contradiction in [*first, *second]:
        key = (contradiction.type, tuple(sorted(contradiction.evidence_ids)), contradiction.description)
        if key not in seen:
            seen.add(key)
            merged.append(contradiction)
    return merged


def _detect_deployment_time_conflicts(evidence_packet: EvidencePacket) -> list[Contradiction]:
    seen: dict[str, tuple[str, str]] = {}
    conflicts = []
    pattern = re.compile(r"\b(DEP-\d+)\b.*?\b(\d{1,2}:\d{2})\b", re.IGNORECASE)
    for evidence in evidence_packet.evidence:
        for deployment, time_value in pattern.findall(evidence.text):
            key = deployment.upper()
            if key in seen and seen[key][0] != time_value:
                conflicts.append(
                    Contradiction(
                        type="timeline_conflict",
                        description=f"{key} has conflicting deployment times: {seen[key][0]} and {time_value}.",
                        evidence_ids=[seen[key][1], evidence.evidence_id],
                        claims=[seen[key][0], time_value],
                        sources=[_source(evidence)],
                    )
                )
            else:
                seen[key] = (time_value, evidence.evidence_id)
    return conflicts


def _detect_database_saturation_conflicts(evidence_packet: EvidencePacket) -> list[Contradiction]:
    positive = []
    negative = []
    for evidence in evidence_packet.evidence:
        text = evidence.text.lower()
        if "connection pool exhaustion" in text or "database saturation" in text:
            positive.append(evidence)
        if "no database saturation" in text or "no connection pool exhaustion" in text:
            negative.append(evidence)
    return [
        Contradiction(
            type="conflicting_claims",
            description="Evidence contains both database saturation/exhaustion claims and denials.",
            evidence_ids=[first.evidence_id, second.evidence_id],
            claims=[first.text[:160], second.text[:160]],
            sources=[_source(first), _source(second)],
        )
        for first in positive
        for second in negative
        if first.evidence_id != second.evidence_id
    ]


def _source(evidence: EvidenceReference) -> dict[str, Any]:
    return {
        "evidence_id": evidence.evidence_id,
        "document_id": evidence.document_id,
        "chunk_id": evidence.chunk_id,
        "document_type": evidence.metadata.get("document_type"),
    }
