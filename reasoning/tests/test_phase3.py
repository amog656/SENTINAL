from __future__ import annotations

import pytest

from app.agent.investigator import Investigator
from app.config import DATA_PATH
from app.llm.cache import DiskLLMCache
from app.llm.client import CachedLLMClient
from app.models.schemas import Evidence, InvestigationGoal, InvestigationState
from app.reasoning.contradiction import classify_guidance_pair, detect_contradictions, resolve_contradiction
from app.reasoning.sufficiency import assess_sufficiency
from app.retrieval.mock_client import MockRetrievalClient


TEST_INPUT_B = "For Service A latency troubleshooting, should we restart the service during a dependency failure? Check the guidance."


def evidence(document_id: str, claim: str, *, supersedes: str | None = None, service: str = "service-a", version: str = "v1") -> Evidence:
    return Evidence(document_id=document_id, document_type="troubleshooting_guide", claims=[claim], facts={
        "service": service, "version": version, "date": "2026-01-01", "supersedes": supersedes,
    })


def test_detector_classifies_direct_contextual_apparent_and_no_contradiction():
    generic = evidence("A", "Restart Service A when latency remains high.")
    dependency_stop = evidence("B", "Do not restart Service A during dependency failures; check dependency health first.")
    direct_stop = evidence("C", "Do not restart Service A when latency remains high.")
    same_action_other_context = evidence("D", "Restart Service A during scheduled maintenance.")
    unrelated = evidence("E", "Check dashboard health before taking action.")
    assert classify_guidance_pair(generic, dependency_stop, generic.claims[0], dependency_stop.claims[0])["type"] == "contextual"
    assert classify_guidance_pair(generic, direct_stop, generic.claims[0], direct_stop.claims[0])["type"] == "direct"
    assert classify_guidance_pair(generic, same_action_other_context, generic.claims[0], same_action_other_context.claims[0])["type"] == "apparent"
    assert classify_guidance_pair(generic, unrelated, generic.claims[0], unrelated.claims[0])["type"] == "no_contradiction"


def test_contextual_guidance_resolves_without_newest_document_wins():
    old = evidence("OLD", "Restart Service A when latency remains high.", version="v1")
    newer = evidence("NEW", "Do not restart Service A during dependency failures; check dependency health first.", supersedes="OLD", version="v3")
    finding = detect_contradictions([old, newer])[0]
    resolved = resolve_contradiction(finding, {"OLD": old, "NEW": newer})
    assert resolved["resolution_attempted"] is True
    assert resolved["resolution_status"] == "resolved"
    assert resolved["rejected_document_id"] is None
    assert "different contexts" in resolved["reason"]
    assert set(resolved["source_document_ids"]) == {"OLD", "NEW"}


def test_investigated_but_unresolved_and_unattempted_are_distinct():
    left = evidence("L", "Restart Service A when latency remains high.")
    right = evidence("R", "Do not restart Service A when latency remains high.")
    finding = detect_contradictions([left, right])[0]
    unresolved = resolve_contradiction(finding, {"L": left, "R": right})
    unattempted = resolve_contradiction(finding, {"L": left, "R": right}, attempt=False)
    assert (unresolved["resolution_attempted"], unresolved["resolution_status"]) == (True, "unresolved")
    assert (unattempted["resolution_attempted"], unattempted["resolution_status"]) == (False, "unresolved")

    # An investigated-but-inconclusive conflict is reportable uncertainty, not
    # an automatic top-level insufficiency result.
    state = InvestigationState(investigation_id="conflict", original_question="conflict", loop_termination_reason="goals_covered")
    state.goals = [InvestigationGoal(id="guidance_conflict", description="Investigate guidance", status="supported")]
    state.resolved_conflicts = [unresolved]
    assert assess_sufficiency(state)[0] == "evidence_supported"


@pytest.mark.asyncio
async def test_input_b_is_retrieved_detected_resolved_and_streamed(tmp_path):
    investigator = Investigator(MockRetrievalClient(DATA_PATH), CachedLLMClient(DiskLLMCache(tmp_path)))
    state, events = investigator.new_state(TEST_INPUT_B), []

    async def sink(event):
        events.append(event)

    report = await investigator.run(state, sink)
    assert {"GUIDE-12", "GUIDE-41"}.issubset(state.documents_seen)
    assert state.contradictions[0]["type"] == "contextual"
    resolution = state.resolved_conflicts[0]
    assert resolution["resolution_status"] == "resolved"
    assert resolution["resolution_attempted"] is True
    assert {"GUIDE-12", "GUIDE-41"}.issubset(report["source_document_ids"])
    types = [event.step_type for event in events]
    assert "contradiction_found" in types and "conflict_resolved" in types
