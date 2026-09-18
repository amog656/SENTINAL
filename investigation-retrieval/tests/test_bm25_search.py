import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.ingestion.service import ingest_and_persist_document
from app.search import bm25
from app.storage.models import Base


def make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def test_tokenizer_preserves_technical_identifiers():
    tokens = bm25.tokenize("INC-1042 DEP-882 v2.8.1 orders-api payment-service")

    assert "inc-1042" in tokens
    assert "dep-882" in tokens
    assert "v2.8.1" in tokens
    assert "orders-api" in tokens
    assert "payment-service" in tokens


def test_bm25_exact_identifier_retrieval(tmp_path):
    db = make_session()
    incident = tmp_path / "incident.json"
    incident.write_text(
        '{"document_id":"INC-1042","type":"incident_report","text":"orders-api degraded after DEP-882 on v2.8.1"}',
        encoding="utf-8",
    )
    generic = tmp_path / "generic.txt"
    generic.write_text("Deployment guidance for services and latency investigations.", encoding="utf-8")
    ingest_and_persist_document(str(incident), db=db, filename=incident.name)
    ingest_and_persist_document(str(generic), db=db, filename=generic.name)

    bm25.rebuild_bm25_index(db)
    results = bm25.search_bm25("DEP-882", limit=5, db=db)

    assert results
    assert results[0]["document_id"] == "INC-1042"
    assert results[0]["score"] > 0


def test_bm25_case_insensitive_keyword_search(tmp_path):
    db = make_session()
    report = tmp_path / "report.txt"
    report.write_text("Root Cause: PostgreSQL connection pool exhaustion caused latency.", encoding="utf-8")
    ingest_and_persist_document(str(report), db=db, filename=report.name)

    bm25.rebuild_bm25_index(db)
    results = bm25.search_bm25("postgresql CONNECTION", limit=5, db=db)

    assert len(results) == 1
    assert "PostgreSQL" in results[0]["text"]


def test_bm25_metadata_filtering(tmp_path):
    db = make_session()
    orders = tmp_path / "orders.txt"
    orders.write_text("orders-api PostgreSQL timeout", encoding="utf-8")
    payment = tmp_path / "payment.txt"
    payment.write_text("payment-service PostgreSQL timeout", encoding="utf-8")
    ingest_and_persist_document(str(orders), db=db, filename=orders.name)
    ingest_and_persist_document(str(payment), db=db, filename=payment.name)

    bm25.rebuild_bm25_index(db)
    results = bm25.search_bm25("PostgreSQL timeout", limit=5, filters={"services": ["orders-api"]}, db=db)

    assert len(results) == 1
    assert results[0]["metadata"]["services"] == ["orders-api"]


def test_bm25_rejects_empty_query():
    with pytest.raises(ValueError, match="query cannot be empty"):
        bm25.search_bm25("   ")


def test_bm25_result_limit_and_deterministic_order(tmp_path):
    db = make_session()
    for index in range(3):
        path = tmp_path / f"incident-{index}.txt"
        path.write_text(f"orders-api latency marker-{index}", encoding="utf-8")
        ingest_and_persist_document(str(path), db=db, filename=path.name)

    bm25.rebuild_bm25_index(db)
    first = bm25.search_bm25("orders-api latency", limit=2, db=db)
    second = bm25.search_bm25("orders-api latency", limit=2, db=db)

    assert len(first) == 2
    assert [result["chunk_id"] for result in first] == [result["chunk_id"] for result in second]
