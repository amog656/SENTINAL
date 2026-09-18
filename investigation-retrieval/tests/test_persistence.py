from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.ingestion.service import ingest_and_persist_document
from app.storage.documents import get_document_by_id
from app.storage.models import Base, StoredChunk, StoredDocument


def make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def test_uploaded_document_and_chunks_are_persisted(tmp_path):
    db = make_session()
    file_path = tmp_path / "INC-1042_report.txt"
    file_path.write_text("orders-api v2.8.1 PostgreSQL latency on 2026-09-14. " * 80, encoding="utf-8")

    stored = ingest_and_persist_document(str(file_path), db=db, filename=file_path.name)

    assert db.query(StoredDocument).count() == 1
    assert db.query(StoredChunk).count() >= 1
    assert stored.services == ["orders-api"]
    assert stored.versions == ["v2.8.1"]
    assert stored.dates == ["2026-09-14"]


def test_duplicate_upload_is_idempotent(tmp_path):
    db = make_session()
    file_path = tmp_path / "incident.json"
    file_path.write_text(
        '{"document_id":"INC-1042","type":"incident_report","text":"orders-api latency DEP-882"}',
        encoding="utf-8",
    )

    first = ingest_and_persist_document(str(file_path), db=db, filename=file_path.name)
    second = ingest_and_persist_document(str(file_path), db=db, filename=file_path.name)

    assert first.document_id == second.document_id == "INC-1042"
    assert db.query(StoredDocument).count() == 1
    assert get_document_by_id(db, "INC-1042") is not None
