import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.storage.models import Base
from app.storage.postgres import get_db


@pytest.fixture
def db_override():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_upload_txt_endpoint_returns_metadata(db_override, monkeypatch):
    indexed = {}
    monkeypatch.setattr("app.search.indexing.encode_texts", lambda texts: [[0.1, 0.2] for _ in texts])
    monkeypatch.setattr("app.search.indexing.delete_document_vectors", lambda document_id: indexed.setdefault("deleted", document_id))
    monkeypatch.setattr(
        "app.search.indexing.upsert_chunk_vectors",
        lambda document, chunks, embeddings: indexed.setdefault("upserted", len(list(chunks))),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/upload",
            files={
                "file": (
                    "INC-1042_report.txt",
                    b"orders-api latency v2.8.1 on 2026-09-14 after DEP-882",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["document"]["filename"] == "INC-1042_report.txt"
    assert data["document"]["document_type"] == "incident_report"
    assert data["document"]["services"] == ["orders-api"]
    assert data["document"]["versions"] == ["v2.8.1"]
    assert data["document"]["dates"] == ["2026-09-14"]
    assert data["document"]["deployment_ids"] == ["DEP-882"]
    assert data["document"]["chunk_count"] == 1
    assert data["document"]["vector_indexed"] is True
    assert data["document"]["indexed_chunks"] == 1
    assert indexed["deleted"] == data["document"]["document_id"]
    assert indexed["upserted"] == 1


@pytest.mark.asyncio
async def test_get_document_endpoint(db_override, monkeypatch):
    monkeypatch.setattr(
        "app.api.upload.index_document_chunks",
        lambda document: {"vector_indexed": False, "indexed_chunks": 0, "indexing_error": "mocked"},
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        upload_response = await client.post(
            "/upload",
            files={
                "file": (
                    "incident.json",
                    b'{"document_id":"INC-1042","type":"incident_report","text":"orders-api PostgreSQL latency DEP-882"}',
                    "application/json",
                )
            },
        )
        document_id = upload_response.json()["document"]["document_id"]
        response = await client.get(f"/documents/{document_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["document"]["document_id"] == "INC-1042"
    assert data["document"]["document_type"] == "incident_report"
    assert data["document"]["chunks"][0]["chunk_id"] == "INC-1042-C001"


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_extension(db_override):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/upload",
            files={"file": ("notes.csv", b"service,orders-api", "text/csv")},
        )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_rejects_empty_file(db_override):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/upload",
            files={"file": ("empty.txt", b"", "text/plain")},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty"
