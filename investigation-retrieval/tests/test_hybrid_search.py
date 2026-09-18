import pytest

from app.search.hybrid import hybrid_search


def result(chunk_id, score, text="evidence", services=None):
    return {
        "chunk_id": chunk_id,
        "document_id": chunk_id.split("-C")[0],
        "score": score,
        "text": text,
        "section": "Root Cause",
        "metadata": {"services": services or ["orders-api"]},
    }


def test_hybrid_merges_duplicate_chunks_and_preserves_provenance():
    results = hybrid_search(
        "DEP-882",
        limit=10,
        bm25_searcher=lambda query, limit, filters=None: [result("C1", 8.0), result("C2", 4.0)],
        semantic_searcher=lambda query, limit, filters=None: [result("C1", 0.9), result("C3", 0.7)],
    )

    by_id = {item["chunk_id"]: item for item in results}
    assert set(by_id) == {"C1", "C2", "C3"}
    assert by_id["C1"]["retrieved_by"] == ["bm25", "vector"]
    assert by_id["C2"]["retrieved_by"] == ["bm25"]
    assert by_id["C3"]["retrieved_by"] == ["vector"]


def test_hybrid_score_normalization_math(monkeypatch):
    monkeypatch.setenv("HYBRID_BM25_WEIGHT", "0.5")
    monkeypatch.setenv("HYBRID_SEMANTIC_WEIGHT", "0.5")

    results = hybrid_search(
        "latency",
        limit=3,
        bm25_searcher=lambda query, limit, filters=None: [result("C1", 10.0), result("C2", 5.0)],
        semantic_searcher=lambda query, limit, filters=None: [result("C1", 0.8), result("C3", 0.6)],
    )
    by_id = {item["chunk_id"]: item for item in results}

    assert by_id["C1"]["normalized_bm25_score"] == 1.0
    assert by_id["C1"]["normalized_semantic_score"] == 1.0
    assert by_id["C1"]["hybrid_score"] == 1.0
    assert by_id["C2"]["hybrid_score"] == 0.0
    assert by_id["C3"]["hybrid_score"] == 0.0


def test_hybrid_exact_identifier_gets_bm25_contribution():
    results = hybrid_search(
        "DEP-882",
        limit=5,
        bm25_searcher=lambda query, limit, filters=None: [result("INC-1042-C001", 7.5, text="DEP-882 orders-api")],
        semantic_searcher=lambda query, limit, filters=None: [],
    )

    assert results[0]["chunk_id"] == "INC-1042-C001"
    assert results[0]["bm25_score"] == 7.5
    assert results[0]["normalized_bm25_score"] == 1.0
    assert results[0]["semantic_score"] is None


def test_hybrid_preserves_semantic_only_results():
    results = hybrid_search(
        "Why did the service become slow?",
        limit=5,
        bm25_searcher=lambda query, limit, filters=None: [],
        semantic_searcher=lambda query, limit, filters=None: [result("INC-1042-C003", 0.86)],
    )

    assert results[0]["chunk_id"] == "INC-1042-C003"
    assert results[0]["retrieved_by"] == ["vector"]
    assert results[0]["normalized_semantic_score"] == 1.0


def test_hybrid_passes_metadata_filters_to_both_searchers():
    seen = []

    def fake_bm25(query, limit, filters=None):
        seen.append(("bm25", filters))
        return [result("C1", 4.0)]

    def fake_semantic(query, limit, filters=None):
        seen.append(("semantic", filters))
        return []

    hybrid_search("connection pool", filters={"services": ["orders-api"]}, bm25_searcher=fake_bm25, semantic_searcher=fake_semantic)

    assert seen == [
        ("bm25", {"services": ["orders-api"]}),
        ("semantic", {"services": ["orders-api"]}),
    ]


def test_hybrid_deterministic_tie_break_by_chunk_id():
    results = hybrid_search(
        "latency",
        limit=10,
        bm25_searcher=lambda query, limit, filters=None: [result("C2", 1.0), result("C1", 1.0)],
        semantic_searcher=lambda query, limit, filters=None: [],
    )

    assert [item["chunk_id"] for item in results] == ["C1", "C2"]


def test_hybrid_rejects_empty_query():
    with pytest.raises(ValueError, match="query cannot be empty"):
        hybrid_search("  ")
