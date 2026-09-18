from typing import Any, Dict, Optional

from app.embeddings.encoder import encode_text
from app.storage.qdrant import search_vectors


def semantic_search(query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> list[dict]:
    if not query or not query.strip():
        raise ValueError("query cannot be empty")
    if limit < 1 or limit > 50:
        raise ValueError("limit must be between 1 and 50")

    query_vector = encode_text(query.strip())
    points = search_vectors(query_vector, limit=limit, filters=filters)

    results = []
    for point in points:
        payload = point.payload or {}
        metadata = {
            "document_type": payload.get("document_type"),
            "services": payload.get("services", []),
            "versions": payload.get("versions", []),
            "dates": payload.get("dates", []),
            "incident_ids": payload.get("incident_ids", []),
            "deployment_ids": payload.get("deployment_ids", []),
            "technical_entities": payload.get("technical_entities", []),
        }
        results.append(
            {
                "chunk_id": payload.get("chunk_id"),
                "document_id": payload.get("document_id"),
                "score": point.score,
                "text": payload.get("text", ""),
                "section": payload.get("section"),
                "metadata": metadata,
            }
        )
    return results
