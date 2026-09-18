import pytest
from httpx import ASGITransport, AsyncClient

from app.api import search as search_api
from app.main import app


def ranked_result():
    return {
        "chunk_id": "INC-1042-C003",
        "document_id": "INC-1042",
        "chunk_index": 3,
        "final_score": 0.87,
        "score_breakdown": {
            "semantic": 0.91,
            "bm25": 0.82,
            "keyword": 1.0,
            "metadata": 1.0,
            "temporal": 0.72,
        },
        "retrieved_by": ["bm25", "vector"],
        "text": "Connection pool exhaustion caused latency.",
        "section": "Root Cause",
        "metadata": {"services": ["orders-api"]},
        "bm25_score": 8.42,
        "semantic_score": 0.86,
    }


@pytest.mark.asyncio
async def test_ranked_search_endpoint_with_mocked_hybrid(monkeypatch):
    monkeypatch.setattr(
        search_api,
        "hybrid_search",
        lambda query, limit, filters=None, bm25_searcher=None: [
            {
                "chunk_id": "INC-1042-C003",
                "document_id": "INC-1042",
                "chunk_index": 3,
                "bm25_score": 8.42,
                "semantic_score": 0.86,
                "text": "orders-api DEP-882 connection pool exhaustion on 2026-09-14",
                "section": "Root Cause",
                "metadata": {
                    "services": ["orders-api"],
                    "deployment_ids": ["DEP-882"],
                    "dates": ["2026-09-14"],
                    "technical_entities": ["connection pool"],
                },
                "retrieved_by": ["bm25", "vector"],
            }
        ],
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/ranked-search",
            json={"query": "Why did orders-api become slow after DEP-882 on 2026-09-14?", "limit": 5},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["score_breakdown"]["keyword"] >= 0.75
    assert data["results"][0]["retrieved_by"] == ["bm25", "vector"]
    assert data["results"][0]["bm25_score"] == 8.42
    assert data["results"][0]["semantic_score"] == 0.86


@pytest.mark.asyncio
async def test_ranked_search_endpoint_rejects_empty_query():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/ranked-search", json={"query": "   "})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_ranked_search_endpoint_rejects_invalid_limits():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        zero_response = await client.post("/ranked-search", json={"query": "latency", "limit": 0})
        large_response = await client.post("/ranked-search", json={"query": "latency", "limit": 51})

    assert zero_response.status_code == 422
    assert large_response.status_code == 422


@pytest.mark.asyncio
async def test_ranked_search_endpoint_returns_no_candidates(monkeypatch):
    monkeypatch.setattr(search_api, "hybrid_search", lambda query, limit, filters=None, bm25_searcher=None: [])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/ranked-search", json={"query": "latency", "limit": 5})

    assert response.status_code == 200
    assert response.json()["results"] == []


@pytest.mark.asyncio
async def test_ranked_search_endpoint_deduplicates_duplicate_candidates(monkeypatch):
    monkeypatch.setattr(
        search_api,
        "hybrid_search",
        lambda query, limit, filters=None, bm25_searcher=None: [
            {"chunk_id": "C1", "document_id": "D", "bm25_score": 5, "text": "latency", "metadata": {}, "retrieved_by": ["bm25"]},
            {"chunk_id": "C1", "document_id": "D", "semantic_score": 0.8, "text": "latency", "metadata": {}, "retrieved_by": ["vector"]},
        ],
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/ranked-search", json={"query": "latency", "limit": 5})

    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["retrieved_by"] == ["bm25", "vector"]


@pytest.mark.asyncio
async def test_ranked_search_openapi_contains_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/openapi.json")

    assert "/ranked-search" in response.json()["paths"]
