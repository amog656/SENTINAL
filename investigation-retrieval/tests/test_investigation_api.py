import pytest
from httpx import ASGITransport, AsyncClient

from app.api import search as search_api
from app.main import app


def result(chunk_id, text, metadata=None):
    return {
        "chunk_id": chunk_id,
        "document_id": "DOC",
        "chunk_index": 1,
        "text": text,
        "section": "Root Cause",
        "final_score": 0.8,
        "score_breakdown": {"semantic": 0.8},
        "retrieved_by": ["bm25"],
        "metadata": metadata or {},
        "bm25_score": 4.0,
        "semantic_score": 0.7,
    }


@pytest.mark.asyncio
async def test_investigation_search_endpoint_mocked_multi_hop(monkeypatch):
    responses = {
        "Why did orders-api become slow after DEP-882?": [result("C1", "orders-api DEP-882", {"deployment_ids": ["DEP-882"]})],
        "DEP-882": [result("C2", "DEP-882 v2.8.1", {"versions": ["v2.8.1"]})],
    }
    monkeypatch.setattr(search_api, "ranked_search_results", lambda query, limit, db: responses.get(query, []))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/investigation-search",
            json={"query": "Why did orders-api become slow after DEP-882?", "max_hops": 1, "results_per_hop": 2},
        )

    assert response.status_code == 200
    data = response.json()
    assert {item["chunk_id"] for item in data["evidence"]} == {"C1", "C2"}
    c2 = next(item for item in data["evidence"] if item["chunk_id"] == "C2")
    assert c2["hop"] == 1
    assert c2["trigger_clue"]["value"] == "DEP-882"
    assert data["relationships"][0]["relationship"] == "triggered_by_clue"


@pytest.mark.asyncio
async def test_investigation_search_rejects_empty_query():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/investigation-search", json={"query": "   "})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_investigation_search_rejects_invalid_max_hops_and_results_per_hop():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        max_hops_response = await client.post("/investigation-search", json={"query": "latency", "max_hops": 0})
        limit_response = await client.post("/investigation-search", json={"query": "latency", "results_per_hop": 51})

    assert max_hops_response.status_code == 422
    assert limit_response.status_code == 422


@pytest.mark.asyncio
async def test_investigation_search_early_termination_and_determinism(monkeypatch):
    monkeypatch.setattr(search_api, "ranked_search_results", lambda query, limit, db: [result("C1", "plain text")])

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post("/investigation-search", json={"query": "plain", "max_hops": 3, "results_per_hop": 1})
        second = await client.post("/investigation-search", json={"query": "plain", "max_hops": 3, "results_per_hop": 1})

    assert first.status_code == 200
    assert first.json() == second.json()
    assert first.json()["summary"]["follow_up_searches"] == 0


@pytest.mark.asyncio
async def test_investigation_search_openapi_contains_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/openapi.json")

    assert "/investigation-search" in response.json()["paths"]
