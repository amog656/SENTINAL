import math
from types import SimpleNamespace

import pytest

from app.ranking import evidence


def candidate(
    chunk_id,
    bm25_score=None,
    semantic_score=None,
    text="orders-api DEP-882 latency on 2026-09-14",
    metadata=None,
    retrieved_by=None,
):
    return {
        "chunk_id": chunk_id,
        "document_id": chunk_id.split("-C")[0] if "-C" in chunk_id else "DOC",
        "chunk_index": 1,
        "text": text,
        "section": "Root Cause",
        "bm25_score": bm25_score,
        "semantic_score": semantic_score,
        "retrieved_by": retrieved_by or ["bm25"],
        "metadata": metadata
        or {
            "services": ["orders-api"],
            "versions": ["v2.8.1"],
            "dates": ["2026-09-14"],
            "document_type": "incident_report",
            "incident_ids": ["INC-1042"],
            "deployment_ids": ["DEP-882"],
            "technical_entities": ["latency"],
        },
    }


def test_normalize_scores_handles_min_max_single_equal_zero_missing_negative_and_invalid():
    assert evidence.normalize_scores([2, 4, 6]) == [0.0, 0.5, 1.0]
    assert evidence.normalize_scores([8]) == [1.0]
    assert evidence.normalize_scores([5, 5]) == [1.0, 1.0]
    assert evidence.normalize_scores([0, 0]) == [0.0, 0.0]
    assert evidence.normalize_scores([None, float("nan"), float("inf")]) == [0.0, 0.0, 0.0]
    assert evidence.normalize_scores([-5, 0, 5]) == [0.0, 0.5, 1.0]
    assert all(math.isfinite(value) for value in evidence.normalize_scores([None, -1, 1]))


def test_keyword_score_rewards_exact_identifiers():
    terms = evidence.meaningful_query_terms("orders-api DEP-882 v2.8.1 INC-1042")
    score = evidence.keyword_score(
        terms,
        candidate(
            "INC-1042-C001",
            text="INC-1042 orders-api v2.8.1 regressed immediately after DEP-882.",
        ),
    )

    assert "orders-api" in terms
    assert "dep-882" in terms
    assert "v2.8.1" in terms
    assert "inc-1042" in terms
    assert score == 1.0


def test_metadata_score_matches_service_version_incident_deployment_and_technical_entity():
    query_metadata = evidence._query_metadata(
        "INC-1042 DEP-882 orders-api v2.8.1 PostgreSQL connection pool latency"
    )
    score = evidence.metadata_score(
        query_metadata,
        {
            "services": ["orders-api"],
            "versions": ["v2.8.1"],
            "incident_ids": ["INC-1042"],
            "deployment_ids": ["DEP-882"],
            "technical_entities": ["postgresql", "connection pool", "latency"],
            "document_type": "incident_report",
        },
    )

    assert score == 1.0


def test_temporal_score_same_nearby_distant_and_missing_dates():
    query_dates = evidence._date_values(["2026-09-14"])

    assert evidence.temporal_score(query_dates, ["2026-09-14"]) == 1.0
    assert evidence.temporal_score(query_dates, ["2026-09-15"]) > evidence.temporal_score(query_dates, ["2026-12-14"])
    assert evidence.temporal_score(query_dates, []) == 0.0


def test_rank_evidence_deduplicates_and_preserves_provenance():
    candidates = [
        candidate("C1", bm25_score=8.4, retrieved_by=["bm25"]),
        candidate("C2", bm25_score=4.0, retrieved_by=["bm25"]),
        candidate("C1", semantic_score=0.87, retrieved_by=["vector"]),
        candidate("C3", semantic_score=0.65, retrieved_by=["vector"]),
    ]

    results = evidence.rank_evidence("orders-api DEP-882", candidates, limit=10)
    by_id = {result["chunk_id"]: result for result in results}

    assert set(by_id) == {"C1", "C2", "C3"}
    assert by_id["C1"]["bm25_score"] == 8.4
    assert by_id["C1"]["semantic_score"] == 0.87
    assert by_id["C1"]["retrieved_by"] == ["bm25", "vector"]


def test_rank_evidence_formula_with_controlled_scores(monkeypatch):
    candidates = [
        candidate("C1", bm25_score=10, semantic_score=0.9),
        candidate("C2", bm25_score=5, semantic_score=0.5, text="unrelated", metadata={}),
    ]

    result = evidence.rank_evidence("orders-api DEP-882 2026-09-14", candidates, limit=2)[0]

    assert result["chunk_id"] == "C1"
    assert result["score_breakdown"] == {
        "semantic": 1.0,
        "bm25": 1.0,
        "keyword": 1.0,
        "metadata": 1.0,
        "temporal": 1.0,
    }
    assert result["final_score"] == 1.0


def test_rank_evidence_orders_by_score_then_chunk_id_and_is_deterministic():
    candidates = [
        candidate("C2", bm25_score=1, semantic_score=1, text="same", metadata={}),
        candidate("C1", bm25_score=1, semantic_score=1, text="same", metadata={}),
    ]

    first = evidence.rank_evidence("same", candidates, limit=10)
    second = evidence.rank_evidence("same", candidates, limit=10)

    assert [result["chunk_id"] for result in first] == ["C1", "C2"]
    assert first == second


def test_ranking_weight_validation_rejects_all_zero(monkeypatch):
    monkeypatch.setattr(
        evidence,
        "get_settings",
        lambda: SimpleNamespace(
            RANK_SEMANTIC_WEIGHT=0,
            RANK_BM25_WEIGHT=0,
            RANK_KEYWORD_WEIGHT=0,
            RANK_METADATA_WEIGHT=0,
            RANK_TEMPORAL_WEIGHT=0,
        ),
    )

    with pytest.raises(ValueError, match="At least one ranking weight"):
        evidence.ranking_weights()
