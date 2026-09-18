import pytest
from httpx import ASGITransport, AsyncClient

from app.api import search as search_api
from app.agent.models import AgentResult
from app.agent.reasoning import build_evidence_packet, unavailable_analysis, validation_failed_analysis
from app.investigation.models import EvidenceItem, InvestigationResult
from app.main import app


def investigation():
    return InvestigationResult(
        query="Why did orders-api slow?",
        max_hops=1,
        hops_completed=0,
        evidence=[EvidenceItem(chunk_id="C1", document_id="D1", text="orders-api latency", final_score=0.9, retrieval_query="root")],
        clues=[],
        relationships=[],
        hops=[],
        trace=[],
        summary={"documents_examined": 1, "chunks_examined": 1, "hops_completed": 0},
    )


@pytest.mark.asyncio
async def test_investigate_endpoint_with_mocked_agent(monkeypatch):
    inv = investigation()
    packet = build_evidence_packet(inv, max_evidence=20)
    analysis = unavailable_analysis()
    analysis.status = "completed"
    analysis.summary = "Mocked."
    analysis.confidence = "medium"
    analysis.evidence_coverage = 1.0

    monkeypatch.setattr(
        search_api,
        "run_reasoning_agent",
        lambda **kwargs: AgentResult(status="completed", query=kwargs["query"], analysis=analysis, investigation=inv, evidence_packet=packet),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/investigate", json={"query": "Why did orders-api slow?", "max_hops": 1, "results_per_hop": 1})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["analysis"]["evidence_coverage"] == 1.0


@pytest.mark.asyncio
async def test_investigate_endpoint_returns_503_for_llm_unavailable(monkeypatch):
    inv = investigation()
    packet = build_evidence_packet(inv, max_evidence=20)
    monkeypatch.setattr(
        search_api,
        "run_reasoning_agent",
        lambda **kwargs: AgentResult(
            status="llm_unavailable",
            query=kwargs["query"],
            analysis=unavailable_analysis(),
            investigation=inv,
            evidence_packet=packet,
            message="Investigation retrieval completed, but no reasoning provider is configured.",
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/investigate", json={"query": "Why did orders-api slow?"})

    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "llm_unavailable"
    assert response.json()["detail"]["investigation"]["evidence"][0]["chunk_id"] == "C1"


@pytest.mark.asyncio
async def test_investigate_endpoint_returns_422_for_validation_failure(monkeypatch):
    inv = investigation()
    packet = build_evidence_packet(inv, max_evidence=20)
    monkeypatch.setattr(
        search_api,
        "run_reasoning_agent",
        lambda **kwargs: AgentResult(
            status="validation_failed",
            query=kwargs["query"],
            analysis=validation_failed_analysis(["Unknown evidence id referenced: E999"]),
            investigation=inv,
            evidence_packet=packet,
            message="LLM response failed validation.",
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/investigate", json={"query": "Why did orders-api slow?"})

    assert response.status_code == 422
    assert "E999" in response.json()["detail"]["analysis"]["validation_errors"][0]


@pytest.mark.asyncio
async def test_investigate_endpoint_rejects_empty_query_and_is_in_openapi():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        bad_response = await client.post("/investigate", json={"query": "   "})
        openapi = await client.get("/openapi.json")

    assert bad_response.status_code == 422
    assert "/investigate" in openapi.json()["paths"]


@pytest.mark.asyncio
async def test_investigate_endpoint_rejects_invalid_bounds():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        hops_response = await client.post("/investigate", json={"query": "latency", "max_hops": 0})
        results_response = await client.post("/investigate", json={"query": "latency", "results_per_hop": 0})

    assert hops_response.status_code == 422
    assert results_response.status_code == 422


@pytest.mark.asyncio
async def test_openapi_keeps_existing_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/openapi.json")

    paths = response.json()["paths"]
    for path in [
        "/",
        "/health",
        "/upload",
        "/documents/{document_id}",
        "/search",
        "/bm25-search",
        "/hybrid-search",
        "/ranked-search",
        "/investigation-search",
        "/admin/reindex-bm25",
        "/investigate",
    ]:
        assert path in paths
