import pytest
from pydantic import ValidationError

from app.agent.models import Finding, ReasoningAnalysis


def test_valid_reasoning_result_model():
    result = ReasoningAnalysis(
        status="completed",
        summary="Evidence supports latency after deployment.",
        confidence="medium",
        findings=[Finding(claim="orders-api latency increased.", type="observed_fact", evidence_ids=["E1"])],
        conclusion="The evidence supports an investigation conclusion.",
    )

    assert result.status == "completed"
    assert result.confidence == "medium"


def test_invalid_confidence_and_status_are_rejected():
    with pytest.raises(ValidationError):
        ReasoningAnalysis(status="done", confidence="medium")

    with pytest.raises(ValidationError):
        ReasoningAnalysis(status="completed", confidence="certain")


def test_empty_finding_claim_is_rejected():
    with pytest.raises(ValidationError):
        Finding(claim="   ", type="observed_fact", evidence_ids=["E1"])
