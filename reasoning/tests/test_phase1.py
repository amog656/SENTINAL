from __future__ import annotations

import asyncio
import os

import pytest

from app.agent.investigator import Investigator
from app.config import CONFIDENCE_BASE, CONFIDENCE_CROSS_DOCUMENT_CONFIRMATION, CONFIDENCE_PER_DOCUMENT_TYPE, CONFIDENCE_PER_SUPPORTING_DOCUMENT, DATA_PATH
from app.llm.cache import DiskLLMCache
from app.llm.client import CachedLLMClient
from app.models.schemas import Evidence, InvestigationState
from app.reasoning.confidence import calculate_confidence
from app.retrieval.mock_client import MockRetrievalClient


async def run_question(tmp_path, question: str):
    investigator = Investigator(MockRetrievalClient(DATA_PATH), CachedLLMClient(DiskLLMCache(tmp_path)))
    state = investigator.new_state(question)
    events = []
    report = await investigator.run(state, lambda event: _capture(events, event))
    return state, events, report


async def _capture(events, event):
    events.append(event)


@pytest.mark.asyncio
async def test_orders_deployment_investigation_cites_real_retrieved_documents(tmp_path):
    state, events, report = await run_question(tmp_path, "Why did the Orders API become slow on September 16? Was the deployment related?")
    # Phase-1 fields remain the same type; Phase 2 adds PM-211 rather than replacing them.
    assert report["source_document_ids"][:2] == ["INC-1042", "DEP-882"]
    assert "PM-211" in report["source_document_ids"]
    assert report["investigation_steps"][-1]["step_type"] == "report_ready"
    kinds = [event.step_type for event in events]
    assert "plan_created" in kinds and kinds.count("search_issued") >= 2 and "clue_extracted" in kinds


def test_confidence_formula_hand_computed_exactly():
    state = InvestigationState(investigation_id="test", original_question="test")
    state.evidence = [
        Evidence(document_id="A", document_type="incident_report", facts={}, claims=[]),
        Evidence(document_id="B", document_type="deployment_note", facts={}, claims=[]),
    ]
    state.deployments = ["B"]
    score, _ = calculate_confidence(state)
    assert score == round(CONFIDENCE_BASE + 2 * CONFIDENCE_PER_SUPPORTING_DOCUMENT + 2 * CONFIDENCE_PER_DOCUMENT_TYPE + CONFIDENCE_CROSS_DOCUMENT_CONFIRMATION, 2)


@pytest.mark.asyncio
async def test_duplicate_search_prevention(tmp_path):
    state, _, _ = await run_question(tmp_path, "Why did the Orders API become slow?")
    assert len({item.query for item in state.search_history}) == len(state.search_history)


@pytest.mark.asyncio
async def test_demo_mode_replays_without_network(tmp_path, monkeypatch):
    question = "Why did the Orders API become slow on September 16?"
    _, _, first = await run_question(tmp_path, question)  # records deterministic cache entries
    monkeypatch.setenv("DEMO_MODE", "replay")
    state, _, replay = await run_question(tmp_path, question)
    assert replay["source_document_ids"] == first["source_document_ids"]
    assert "PM-211" in replay["source_document_ids"]
    assert state.historical_comparisons and state.timeline
