from __future__ import annotations

import pytest

from app.agent.investigator import Investigator
from app.config import DATA_PATH, MAX_HOPS
from app.llm.cache import DiskLLMCache
from app.llm.client import CachedLLMClient
from app.models.schemas import InvestigationGoal, InvestigationState
from app.reasoning.sufficiency import assess_sufficiency
from app.retrieval.interface import RetrievalClient
from app.retrieval.mock_client import MockRetrievalClient
from tests.test_phase1 import run_question


ORDERS_QUESTION = "Why did the Orders API become slow on September 16? Check whether the deployment was related and whether we have seen this before."


@pytest.mark.asyncio
async def test_orders_investigation_reaches_three_retrieval_hops(tmp_path):
    state, _, _ = await run_question(tmp_path, ORDERS_QUESTION)
    assert len(state.search_history) >= 3
    assert len(state.search_history) <= MAX_HOPS


@pytest.mark.asyncio
async def test_historical_document_is_discovered_via_search(tmp_path):
    state, _, _ = await run_question(tmp_path, ORDERS_QUESTION)
    historical_search = next(record for record in state.search_history if "postmortem" in record.query)
    assert historical_search.result_document_ids == ["PM-211"]
    assert "PM-211" in state.documents_seen


@pytest.mark.asyncio
async def test_historical_comparison_is_similar_but_not_same_root_cause(tmp_path):
    state, events, _ = await run_question(tmp_path, ORDERS_QUESTION)
    comparison = state.historical_comparisons[0]
    assert comparison["comparison_performed"] is True
    assert comparison["similarity"] == "medium"
    assert comparison["similarity_dimensions"] == {"service": "high", "symptom": "high", "root_cause": "unknown"}
    assert comparison["same_root_cause"] == "not_established"
    assert comparison["conclusion"] == "Similar symptom, but same root cause is not established."
    assert "historical_comparison" in [event.step_type for event in events]


@pytest.mark.asyncio
async def test_timeline_is_chronological_and_source_traceable(tmp_path):
    state, events, _ = await run_question(tmp_path, ORDERS_QUESTION)
    assert {event["source"] for event in state.timeline} >= {"DEP-882", "INC-1042"}
    assert state.timeline == sorted(state.timeline, key=lambda event: (event["timestamp"] or event["date"], event["source"]))
    assert next(event for event in state.timeline if event["source"] == "PM-211")["timestamp"] is None
    assert "timeline_updated" in [event.step_type for event in events]


@pytest.mark.asyncio
async def test_deployment_relationship_is_temporal_not_causal(tmp_path):
    state, _, report = await run_question(tmp_path, ORDERS_QUESTION)
    relationship = next(item for item in state.hypotheses if item["relationship_type"] == "temporal_precedence")
    assert relationship["causation"] == "not_established"
    assert "The deployment caused the incident." in report["what_is_not_established"]


@pytest.mark.asyncio
async def test_duplicate_queries_are_prevented_and_reason_is_recorded(tmp_path):
    state, events, _ = await run_question(tmp_path, ORDERS_QUESTION)
    queries = [record.query.casefold().strip() for record in state.search_history]
    assert len(queries) == len(set(queries))
    assert state.loop_termination_reason == "goals_covered"
    sufficiency = next(event for event in events if event.step_type == "sufficiency_checked")
    assert sufficiency.payload["loop_termination_reason"] == "goals_covered"


@pytest.mark.asyncio
async def test_completed_negative_comparison_marks_goal_supported_and_overall_evidence_supported(tmp_path):
    state, _, report = await run_question(tmp_path, ORDERS_QUESTION)
    comparison = state.historical_comparisons[0]
    historical_goal = next(goal for goal in state.goals if goal.id == "historical_comparison")
    assert comparison["comparison_performed"] is True
    assert comparison["same_root_cause"] == "not_established"
    assert historical_goal.status == "supported"
    assert state.loop_termination_reason == "goals_covered"
    assert report["status"] == "evidence_supported"
    assert "Whether the current incident has the same root cause as PM-211." in report["open_questions"]


class CurrentIncidentOnlyRetrieval(RetrievalClient):
    """Minimal fixture: a current incident exists, but no comparable history does."""

    def __init__(self):
        self.current = MockRetrievalClient(DATA_PATH).documents[0]

    async def search(self, query: str, filters=None):
        return [self.current] if query == "orders-api latency incident" else []


@pytest.mark.asyncio
async def test_unperformable_comparison_remains_distinct_from_completed_negative_comparison(tmp_path):
    agent = Investigator(CurrentIncidentOnlyRetrieval(), CachedLLMClient(DiskLLMCache(tmp_path)))
    state = agent.new_state(ORDERS_QUESTION)
    events = []

    async def sink(event):
        events.append(event)

    report = await agent.run(state, sink)
    comparison = state.historical_comparisons[0]
    comparison_goal = next(goal for goal in state.goals if goal.id == "historical_comparison")
    assert comparison["comparison_performed"] is False
    assert comparison_goal.status == "unsupported"  # Targeted history retrieval found no comparable document.
    assert report["status"] == "insufficient_evidence"


@pytest.mark.asyncio
async def test_max_hops_budget_produces_unresolved_when_a_critical_goal_remains(tmp_path, monkeypatch):
    monkeypatch.setattr("app.agent.planner.MAX_HOPS", 2)
    state, _, report = await run_question(tmp_path, ORDERS_QUESTION)
    assert len(state.search_history) == 2
    assert state.loop_termination_reason == "max_hops_reached"
    assert report["status"] == "unresolved"


def test_goal_mapping_distinguishes_unresolved_from_insufficient_evidence():
    unresolved = InvestigationState(investigation_id="u", original_question="u", loop_termination_reason="max_hops_reached")
    unresolved.goals = [InvestigationGoal(id="needed", description="Needed goal", status="not_investigated")]
    assert assess_sufficiency(unresolved)[0] == "unresolved"

    insufficient = InvestigationState(investigation_id="i", original_question="i", loop_termination_reason="no_targetable_goal")
    insufficient.goals = [InvestigationGoal(id="needed", description="Needed goal", status="unresolved")]
    assert assess_sufficiency(insufficient)[0] == "insufficient_evidence"
