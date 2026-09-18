from types import SimpleNamespace

from app.storage.models import StoredChunk, StoredDocument
from app.storage.qdrant import (
    build_filter,
    deterministic_vector_id,
    ensure_collection,
    search_vectors,
    upsert_chunk_vectors,
)


class FakeQdrantClient:
    def __init__(self):
        self.collections = []
        self.created = []
        self.upserted = []
        self.last_search = None

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.collections])

    def create_collection(self, collection_name, vectors_config):
        self.collections.append(collection_name)
        self.created.append((collection_name, vectors_config))

    def upsert(self, collection_name, points):
        self.upserted.append((collection_name, points))

    def query_points(self, collection_name, query, query_filter, limit, with_payload):
        self.last_search = (collection_name, query, query_filter, limit, with_payload)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=0.91,
                    payload={
                        "chunk_id": "INC-1042-C001",
                        "document_id": "INC-1042",
                        "text": "orders-api latency",
                    },
                )
            ]
        )


def make_document_and_chunk():
    document = StoredDocument(
        document_id="INC-1042",
        filename="incident.txt",
        document_type="incident_report",
        source="upload",
        file_size=100,
        character_count=42,
        text="orders-api latency",
        metadata_json={},
        services=["orders-api"],
        versions=["v2.8.1"],
        dates=["2026-09-14"],
        incident_ids=["INC-1042"],
        deployment_ids=["DEP-882"],
        technical_entities=["latency"],
    )
    chunk = StoredChunk(
        chunk_id="INC-1042-C001",
        document_id="INC-1042",
        chunk_index=0,
        text="orders-api latency",
        section="Summary",
        metadata_json={},
    )
    return document, chunk


def test_collection_creation():
    client = FakeQdrantClient()

    ensure_collection(2, collection_name="test_chunks", client=client)

    assert client.collections == ["test_chunks"]
    assert client.created[0][0] == "test_chunks"


def test_upsert_uses_deterministic_ids():
    client = FakeQdrantClient()
    document, chunk = make_document_and_chunk()

    count = upsert_chunk_vectors(document, [chunk], [[0.1, 0.2]], collection_name="test_chunks", client=client)

    assert count == 1
    point = client.upserted[0][1][0]
    assert point.id == deterministic_vector_id("INC-1042-C001")
    assert point.payload["chunk_id"] == "INC-1042-C001"
    assert point.payload["services"] == ["orders-api"]


def test_search_and_metadata_filter():
    client = FakeQdrantClient()
    results = search_vectors(
        [0.1, 0.2],
        limit=5,
        filters={"services": ["orders-api"], "document_type": "incident_report"},
        collection_name="test_chunks",
        client=client,
    )

    assert len(results) == 1
    assert results[0].payload["chunk_id"] == "INC-1042-C001"
    assert client.last_search[3] == 5


def test_build_filter_rejects_unknown_fields():
    try:
        build_filter({"unknown": "value"})
    except ValueError as exc:
        assert "Unsupported filter field" in str(exc)
    else:
        raise AssertionError("Expected unsupported filter to fail")
