import pytest

from app.agent.reasoning import (
    ReasoningValidationError,
    build_evidence_packet,
    calculate_evidence_coverage,
    detect_contradictions,
    validate_reasoning_output,
)
from app.investigation.models import EvidenceItem, EvidenceRelationship, InvestigationResult


def investigation():
    return InvestigationResult(
        query="Why did orders-api slow after DEP-882?",
        max_hops=2,
        hops_completed=1,
        evidence=[
            EvidenceItem(
                chunk_id="INC-1042-C001",
                document_id="INC-1042",
                chunk_index=1,
                text="orders-api experienced latency after DEP-882.",
                final_score=0.91,
                retrieval_query="root",
                metadata={"services": ["orders-api"], "dates": ["2026-09-14"]},
            ),
            EvidenceItem(
                chunk_id="DEP-882-C001",
                document_id="DEP-882",
                chunk_index=1,
                text="DEP-882 deployed v2.8.1 to orders-api on 2026-09-14.",
                final_score=0.88,
                hop=1,
                retrieval_query="DEP-882",
                metadata={"deployment_ids": ["DEP-882"], "versions": ["v2.8.1"], "dates": ["2026-09-14"]},
            ),
        ],
        relationships=[
            EvidenceRelationship(
                from_chunk_id="INC-1042-C001",
                to_chunk_id="DEP-882-C001",
                relationship="triggered_by_clue",
                clue="DEP-882",
                hop=1,
            )
        ],
        clues=[],
        hops=[],
        trace=[],
        summary={"documents_examined": 2, "chunks_examined": 2, "hops_completed": 1},
    )


def test_evidence_packet_is_deterministic_and_uses_evidence_ids():
    first = build_evidence_packet(investigation(), max_evidence=20)
    second = build_evidence_packet(investigation(), max_evidence=20)

    assert first.model_dump() == second.model_dump()
    assert "[E1]" in first.text
    assert first.evidence[0].evidence_id == "E1"
    assert first.relationships[0]["from"] == "E1"
    assert first.relationships[0]["to"] == "E2"


def test_validate_reasoning_accepts_valid_json_and_calculates_coverage():
    packet = build_evidence_packet(investigation(), max_evidence=20)
    raw = """
    ```json
    {
      "status": "completed",
      "summary": "Latency was documented after deployment.",
      "confidence": "medium",
      "findings": [{"claim": "orders-api latency increased.", "type": "observed_fact", "evidence_ids": ["E1"]}],
      "timeline": [{"date": "2026-09-14", "event": "DEP-882 deployed v2.8.1.", "evidence_ids": ["E2"]}],
      "causal_chain": [{"step": "Deployment and latency are temporally associated.", "type": "inference", "evidence_ids": ["E1", "E2"]}],
      "contradictions": [],
      "hypotheses": [],
      "missing_evidence": ["Configuration diff for DEP-882"],
      "conclusion": "Evidence is suggestive but not complete.",
      "evidence_used": ["E1", "E2"]
    }
    ```
    """

    analysis = validate_reasoning_output(raw, packet)

    assert analysis.evidence_coverage == 1.0
    assert analysis.timeline[0].date == "2026-09-14"


def test_unknown_evidence_id_fails_validation():
    packet = build_evidence_packet(investigation(), max_evidence=20)
    raw = '{"status":"completed","summary":"","confidence":"low","findings":[{"claim":"bad","type":"observed_fact","evidence_ids":["E999"]}],"timeline":[],"causal_chain":[],"contradictions":[],"hypotheses":[],"missing_evidence":[],"conclusion":"","evidence_used":["E999"]}'

    with pytest.raises(ReasoningValidationError):
        validate_reasoning_output(raw, packet)


def test_unsupported_factual_claim_fails_validation():
    packet = build_evidence_packet(investigation(), max_evidence=20)
    raw = '{"status":"completed","summary":"","confidence":"low","findings":[{"claim":"Redis caused the outage","type":"observed_fact","evidence_ids":[]}],"timeline":[],"causal_chain":[],"contradictions":[],"hypotheses":[],"missing_evidence":[],"conclusion":"","evidence_used":[]}'

    with pytest.raises(ReasoningValidationError):
        validate_reasoning_output(raw, packet)


def test_inference_with_evidence_is_accepted():
    packet = build_evidence_packet(investigation(), max_evidence=20)
    raw = '{"status":"completed","summary":"","confidence":"medium","findings":[{"claim":"The deployment is temporally associated with latency","type":"inference","evidence_ids":["E1","E2"]}],"timeline":[],"causal_chain":[],"contradictions":[],"hypotheses":[],"missing_evidence":[],"conclusion":"","evidence_used":["E1","E2"]}'

    analysis = validate_reasoning_output(raw, packet)

    assert analysis.findings[0].type == "inference"


def test_contradiction_detection_finds_timeline_and_claim_conflicts():
    investigation_result = investigation()
    investigation_result.evidence.append(
        EvidenceItem(
            chunk_id="DEP-882-C002",
            document_id="DEP-882",
            text="DEP-882 deployment occurred at 10:05. No database saturation was observed.",
            retrieval_query="DEP-882",
            final_score=0.7,
        )
    )
    investigation_result.evidence.append(
        EvidenceItem(
            chunk_id="DEP-882-C003",
            document_id="DEP-882",
            text="DEP-882 deployment occurred at 10:30. PostgreSQL connection pool exhaustion caused latency.",
            retrieval_query="DEP-882",
            final_score=0.6,
        )
    )
    packet = build_evidence_packet(investigation_result, max_evidence=20)

    contradictions = detect_contradictions(packet)

    assert any(item.type == "timeline_conflict" for item in contradictions)
    assert any(item.type == "conflicting_claims" for item in contradictions)


def test_evidence_coverage_is_deterministic():
    packet = build_evidence_packet(investigation(), max_evidence=20)
    raw = '{"status":"completed","summary":"","confidence":"low","findings":[{"claim":"supported","type":"observed_fact","evidence_ids":["E1"]}],"timeline":[],"causal_chain":[],"contradictions":[],"hypotheses":[{"claim":"maybe","type":"hypothesis","evidence_ids":[]}],"missing_evidence":[],"conclusion":"","evidence_used":["E1"]}'
    analysis = validate_reasoning_output(raw, packet)

    assert calculate_evidence_coverage(analysis) == 0.5
