import pytest
from httpx import ASGITransport, AsyncClient

from app.api import search as search_api
from app.main import app


@pytest.mark.asyncio
async def test_search_endpoint(monkeypatch):
    monkeypatch.setattr(
        search_api,
        "semantic_search",
        lambda query, limit, filters=None: [
            {
                "chunk_id": "INC-1042-C001",
                "document_id": "INC-1042",
                "score": 0.92,
                "text": "orders-api latency after deployment",
                "section": "Timeline",
                "metadata": {"services": ["orders-api"]},
            }
        ],
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/search",
            json={"query": "Why did orders-api become slow?", "limit": 5, "filters": {"services": ["orders-api"]}},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Why did orders-api become slow?"
    assert data["results"][0]["chunk_id"] == "INC-1042-C001"


@pytest.mark.asyncio
async def test_search_endpoint_rejects_empty_query():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/search", json={"query": "   "})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_endpoint_rejects_bad_filter():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/search", json={"query": "latency", "filters": {"bad": "value"}})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_bm25_search_endpoint(monkeypatch):
    monkeypatch.setattr(
        search_api,
        "search_bm25",
        lambda query, limit, filters=None, db=None: [
            {
                "chunk_id": "INC-1042-C001",
                "document_id": "INC-1042",
                "score": 8.42,
                "text": "orders-api latency after DEP-882",
                "section": "Timeline",
                "metadata": {"services": ["orders-api"]},
            }
        ],
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/bm25-search", json={"query": "INC-1042 DEP-882", "limit": 5})

    assert response.status_code == 200
    assert response.json()["results"][0]["score"] == 8.42


@pytest.mark.asyncio
async def test_hybrid_search_endpoint(monkeypatch):
    monkeypatch.setattr(
        search_api,
        "hybrid_search",
        lambda query, limit, filters=None, bm25_searcher=None: [
            {
                "chunk_id": "INC-1042-C003",
                "document_id": "INC-1042",
                "hybrid_score": 0.91,
                "bm25_score": 8.42,
                "semantic_score": 0.86,
                "normalized_bm25_score": 1.0,
                "normalized_semantic_score": 0.82,
                "text": "PostgreSQL connection pool exhaustion.",
                "section": "Root Cause",
                "metadata": {"services": ["orders-api"]},
                "retrieved_by": ["bm25", "vector"],
            }
        ],
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/hybrid-search", json={"query": "Why did orders-api become slow?", "limit": 5})

    assert response.status_code == 200
    data = response.json()
    assert data["results"][0]["retrieved_by"] == ["bm25", "vector"]
    assert data["results"][0]["hybrid_score"] == 0.91


@pytest.mark.asyncio
async def test_search_openapi_contains_phase_5_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/openapi.json")

    paths = response.json()["paths"]
    assert "/bm25-search" in paths
    assert "/hybrid-search" in paths
    assert "/admin/reindex-bm25" in paths
