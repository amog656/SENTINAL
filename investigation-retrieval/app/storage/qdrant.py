import uuid
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import get_settings
from app.storage.models import StoredChunk, StoredDocument


VECTOR_NAMESPACE = uuid.UUID("78279d4d-d3c7-4a61-94c2-61ea51f7f99b")


class QdrantIndexError(Exception):
    pass


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url, timeout=2)


def deterministic_vector_id(chunk_id: str) -> str:
    return str(uuid.uuid5(VECTOR_NAMESPACE, chunk_id))


def check_qdrant_connection() -> bool:
    try:
        client = get_qdrant_client()
        client.get_collections()
        return True
    except Exception:
        return False


def ensure_collection(vector_size: int, collection_name: Optional[str] = None, client: Optional[QdrantClient] = None) -> None:
    settings = get_settings()
    collection = collection_name or settings.QDRANT_COLLECTION
    client = client or get_qdrant_client()

    existing = [item.name for item in client.get_collections().collections]
    if collection in existing:
        return

    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )


def build_chunk_payload(chunk: StoredChunk, document: StoredDocument) -> Dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "section": chunk.section,
        "document_type": document.document_type,
        "services": document.services,
        "versions": document.versions,
        "dates": document.dates,
        "incident_ids": document.incident_ids,
        "deployment_ids": document.deployment_ids,
        "technical_entities": document.technical_entities,
    }


def upsert_chunk_vectors(
    document: StoredDocument,
    chunks: Iterable[StoredChunk],
    embeddings: list[list[float]],
    collection_name: Optional[str] = None,
    client: Optional[QdrantClient] = None,
) -> int:
    chunks = list(chunks)
    if not chunks:
        return 0
    if len(chunks) != len(embeddings):
        raise QdrantIndexError("Chunk count and embedding count do not match")

    settings = get_settings()
    collection = collection_name or settings.QDRANT_COLLECTION
    client = client or get_qdrant_client()
    ensure_collection(len(embeddings[0]), collection_name=collection, client=client)

    points = [
        models.PointStruct(
            id=deterministic_vector_id(chunk.chunk_id),
            vector=embedding,
            payload=build_chunk_payload(chunk, document),
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=collection, points=points)
    return len(points)


def build_filter(filters: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
    if not filters:
        return None

    allowed = {"services", "document_type", "versions", "incident_ids", "deployment_ids"}
    conditions = []
    for key, value in filters.items():
        if key not in allowed:
            raise ValueError(f"Unsupported filter field: {key}")
        values = value if isinstance(value, list) else [value]
        if not values or any(not isinstance(item, str) or not item.strip() for item in values):
            raise ValueError(f"Filter '{key}' must contain non-empty string values")
        conditions.append(models.FieldCondition(key=key, match=models.MatchAny(any=values)))

    return models.Filter(must=conditions)


def search_vectors(
    query_vector: list[float],
    limit: int,
    filters: Optional[Dict[str, Any]] = None,
    collection_name: Optional[str] = None,
    client: Optional[QdrantClient] = None,
):
    settings = get_settings()
    collection = collection_name or settings.QDRANT_COLLECTION
    client = client or get_qdrant_client()
    query_filter = build_filter(filters)

    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return response.points

    return client.search(
        collection_name=collection,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )


def delete_document_vectors(document_id: str, collection_name: Optional[str] = None, client: Optional[QdrantClient] = None) -> None:
    settings = get_settings()
    collection = collection_name or settings.QDRANT_COLLECTION
    client = client or get_qdrant_client()
    client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
        ),
    )
