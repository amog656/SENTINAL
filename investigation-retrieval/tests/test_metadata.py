from app.ingestion.metadata import (
    detect_document_type,
    extract_dates,
    extract_deployment_ids,
    extract_incident_ids,
    extract_metadata,
    extract_services,
    extract_technical_entities,
    extract_versions,
)


def test_service_extraction_known_services():
    assert extract_services("Orders-API experienced latency.", ["orders-api"]) == ["orders-api"]


def test_version_extraction_avoids_plain_decimals():
    text = "orders-api v2.8.1 rolled back from 2.8.1 after 10.05 AM and code 500.23"
    assert extract_versions(text) == ["v2.8.1", "2.8.1"]


def test_incident_id_extraction():
    assert extract_incident_ids("inc-1042 and INC-001") == ["INC-1042", "INC-001"]


def test_deployment_id_extraction():
    assert extract_deployment_ids("dep-882 followed DEP-001") == ["DEP-882", "DEP-001"]


def test_date_extraction_normalizes_common_formats():
    text = "2026-09-14, 2026/09/15, 14-09-2026, September 16, 2026, Sep 17, 2026 at 10:05 AM"
    assert extract_dates(text) == [
        "2026-09-14",
        "2026-09-15",
        "2026-09-16",
        "2026-09-17",
    ]


def test_document_type_detection():
    assert detect_document_type("INC-1042_report.pdf", "Impact and timeline") == "incident_report"
    assert detect_document_type("notes.json", "anything", {"type": "deployment_note"}) == "deployment_note"
    assert detect_document_type("random.txt", "misc notes") == "unknown"


def test_technical_entity_extraction():
    text = "PostgreSQL connection pool exhaustion caused latency and HTTP 500 responses."
    entities = extract_technical_entities(text)
    assert "connection pool exhaustion" in entities
    assert "connection pool" in entities
    assert "http 500" in entities
    assert "postgresql" in entities
    assert "latency" in entities


def test_extract_metadata_combines_fields():
    metadata = extract_metadata(
        "INC-1042_report.txt",
        "orders-api v2.8.1 hit PostgreSQL latency on 2026-09-14 after DEP-882.",
    )
    assert metadata.services == ["orders-api"]
    assert metadata.versions == ["v2.8.1"]
    assert metadata.dates == ["2026-09-14"]
    assert metadata.document_type == "incident_report"
    assert metadata.deployment_ids == ["DEP-882"]
    assert "postgresql" in metadata.technical_entities
