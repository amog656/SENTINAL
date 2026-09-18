import json

import pytest

from app.agent.llm import LLMUnavailableError, OpenAICompatibleProvider, create_llm_provider
from app.agent.reasoning import ReasoningValidationError, build_evidence_packet, validate_reasoning_output
from app.config import get_settings
from app.investigation.models import EvidenceItem, InvestigationResult


def three_evidence_investigation():
    return InvestigationResult(
        query="Why did orders-api become slow after DEP-882?",
        max_hops=2,
        hops_completed=1,
        evidence=[
            EvidenceItem(chunk_id="C1", document_id="D1", text="orders-api became slow.", final_score=0.95, retrieval_query="root"),
            EvidenceItem(chunk_id="C2", document_id="D2", text="DEP-882 deployed v2.8.1.", final_score=0.9, retrieval_query="DEP-882", hop=1),
            EvidenceItem(chunk_id="C3", document_id="D3", text="connection pool exhaustion occurred.", final_score=0.85, retrieval_query="connection pool", hop=1),
        ],
        clues=[],
        relationships=[],
        hops=[],
        trace=[],
        summary={"documents_examined": 3, "chunks_examined": 3, "hops_completed": 1},
    )


def reasoning_payload(evidence_ids=None, claim="Evidence-backed finding.", claim_type="observed_fact"):
    evidence_ids = ["E1"] if evidence_ids is None else evidence_ids
    return json.dumps(
        {
            "status": "completed",
            "summary": "Grounded analysis.",
            "confidence": "medium",
            "findings": [{"claim": claim, "type": claim_type, "evidence_ids": evidence_ids}],
            "timeline": [{"date": None, "event": "Undated event from evidence.", "evidence_ids": ["E1"]}],
            "causal_chain": [],
            "contradictions": [],
            "hypotheses": [],
            "missing_evidence": ["Configuration diff for DEP-882"],
            "conclusion": "Conclusion remains evidence-bounded.",
            "evidence_used": evidence_ids,
        }
    )


def test_agent_modules_import_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()

    provider = create_llm_provider()

    with pytest.raises(LLMUnavailableError) as exc:
        provider.generate("system", "user")
    assert "OPENAI_API_KEY" not in str(exc.value)
    get_settings.cache_clear()


def test_openai_provider_initialization_is_lazy(monkeypatch):
    provider = OpenAICompatibleProvider(api_key="secret-test-key", model="test-model")

    assert provider.api_key == "secret-test-key"
    assert provider.model == "test-model"


def test_e1_e2_e3_grounding_succeeds_and_e999_fails():
    packet = build_evidence_packet(three_evidence_investigation(), max_evidence=20)

    analysis = validate_reasoning_output(reasoning_payload(["E1", "E2", "E3"]), packet)
    assert analysis.status == "completed"
    assert analysis.evidence_used == ["E1", "E2", "E3"]

    with pytest.raises(ReasoningValidationError):
        validate_reasoning_output(reasoning_payload(["E999"]), packet)


def test_unsupported_redis_fact_without_evidence_is_rejected():
    packet = build_evidence_packet(three_evidence_investigation(), max_evidence=20)

    with pytest.raises(ReasoningValidationError):
        validate_reasoning_output(reasoning_payload([], "Redis caused the outage."), packet)


def test_temporal_association_inference_is_allowed_with_evidence():
    packet = build_evidence_packet(three_evidence_investigation(), max_evidence=20)

    analysis = validate_reasoning_output(
        reasoning_payload(["E1", "E2"], "DEP-882 is temporally associated with the incident.", "inference"),
        packet,
    )

    assert analysis.findings[0].type == "inference"
