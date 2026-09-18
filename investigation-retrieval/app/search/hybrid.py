from typing import Any, Callable, Dict, Optional

from app.config import get_settings
from app.search.bm25 import search_bm25, validate_search_input
from app.search.semantic import semantic_search


def hybrid_search(
    query: str,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    bm25_searcher: Callable[..., list[dict]] = search_bm25,
    semantic_searcher: Callable[..., list[dict]] = semantic_search,
) -> list[dict]:
    validate_search_input(query, limit)

    candidate_limit = min(get_settings().MAX_SEARCH_LIMIT, max(limit * 3, limit))
    bm25_results = bm25_searcher(query, limit=candidate_limit, filters=filters)
    semantic_results = semantic_searcher(query, limit=candidate_limit, filters=filters)

    normalized_bm25 = _normalize_scores(bm25_results, "score")
    normalized_semantic = _normalize_scores(semantic_results, "score")
    merged: dict[str, dict] = {}

    for result in bm25_results:
        chunk_id = result["chunk_id"]
        merged[chunk_id] = _base_result(result)
        merged[chunk_id]["bm25_score"] = result["score"]
        merged[chunk_id]["normalized_bm25_score"] = normalized_bm25[chunk_id]
        merged[chunk_id]["retrieved_by"].append("bm25")

    for result in semantic_results:
        chunk_id = result["chunk_id"]
        if chunk_id not in merged:
            merged[chunk_id] = _base_result(result)
        merged[chunk_id]["semantic_score"] = result["score"]
        merged[chunk_id]["normalized_semantic_score"] = normalized_semantic[chunk_id]
        if "vector" not in merged[chunk_id]["retrieved_by"]:
            merged[chunk_id]["retrieved_by"].append("vector")

    bm25_weight = get_settings().HYBRID_BM25_WEIGHT
    semantic_weight = get_settings().HYBRID_SEMANTIC_WEIGHT
    for result in merged.values():
        result["bm25_score"] = result.get("bm25_score")
        result["semantic_score"] = result.get("semantic_score")
        result["normalized_bm25_score"] = result.get("normalized_bm25_score", 0.0)
        result["normalized_semantic_score"] = result.get("normalized_semantic_score", 0.0)
        result["hybrid_score"] = (
            bm25_weight * result["normalized_bm25_score"]
            + semantic_weight * result["normalized_semantic_score"]
        )

    results = list(merged.values())
    results.sort(key=lambda result: (-result["hybrid_score"], result["chunk_id"]))
    return results[:limit]


def _normalize_scores(results: list[dict], score_field: str) -> dict[str, float]:
    """Min-max normalize one retrieval method over its returned candidate set.

    If the method returned one result, or all returned scores are equal, each
    meaningful returned candidate receives 1.0 to avoid division by zero.
    Missing candidates in the merged set are treated as 0.0 by hybrid_search.
    """
    if not results:
        return {}

    scores = [float(result[score_field]) for result in results]
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return {result["chunk_id"]: 1.0 for result in results}

    return {
        result["chunk_id"]: (float(result[score_field]) - minimum) / (maximum - minimum)
        for result in results
    }


def _base_result(result: dict) -> dict:
    return {
        "chunk_id": result["chunk_id"],
        "document_id": result["document_id"],
        "chunk_index": result.get("chunk_index"),
        "text": result.get("text", ""),
        "section": result.get("section"),
        "metadata": result.get("metadata", {}),
        "retrieved_by": [],
    }
