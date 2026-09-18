from types import SimpleNamespace

import pytest

from app.search import semantic


def test_semantic_search_returns_results(monkeypatch):
    monkeypatch.setattr(semantic, "encode_text", lambda text: [0.1, 0.2])
    monkeypatch.setattr(
        semantic,
        "search_vectors",
        lambda query_vector, limit, filters=None: [
            SimpleNamespace(
                score=0.84,
                payload={
                    "chunk_id": "INC-1042-C003",
                    "document_id": "INC-1042",
                    "text": "Root cause was PostgreSQL connection pool exhaustion.",
                    "section": "Root Cause",
                    "document_type": "incident_report",
                    "services": ["orders-api"],
                    "versions": ["v2.8.1"],
                    "incident_ids": ["INC-1042"],
                    "deployment_ids": ["DEP-882"],
                    "technical_entities": ["postgresql", "connection pool"],
                },
            )
        ],
    )

    results = semantic.semantic_search("database connection problems", limit=5, filters={"services": ["orders-api"]})

    assert results[0]["chunk_id"] == "INC-1042-C003"
    assert results[0]["score"] == 0.84
    assert results[0]["metadata"]["services"] == ["orders-api"]


def test_semantic_search_rejects_empty_query():
    with pytest.raises(ValueError, match="query cannot be empty"):
        semantic.semantic_search("  ")
