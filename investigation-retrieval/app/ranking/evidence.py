import math
from datetime import date
from typing import Any, Dict, Optional

from app.config import get_settings
from app.ingestion.metadata import extract_dates, extract_metadata
from app.search.bm25 import tokenize, validate_search_input


STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "was",
    "why",
    "did",
    "what",
    "how",
    "when",
    "where",
    "of",
    "to",
    "in",
    "after",
    "before",
    "and",
    "or",
    "for",
    "with",
    "on",
    "became",
    "become",
}

METADATA_FIELDS = ("services", "versions", "incident_ids", "deployment_ids", "technical_entities")


def rank_evidence(query: str, candidates: list[dict], limit: int = 10) -> list[dict]:
    validate_search_input(query, limit)
    merged = _deduplicate_candidates(candidates)
    if not merged:
        return []

    bm25_scores = normalize_scores([candidate.get("bm25_score") for candidate in merged])
    semantic_scores = normalize_scores([candidate.get("semantic_score") for candidate in merged])
    query_terms = meaningful_query_terms(query)
    query_metadata = _query_metadata(query)
    query_dates = _date_values(extract_dates(query))
    weights = ranking_weights()

    ranked = []
    for index, candidate in enumerate(merged):
        keyword = keyword_score(query_terms, candidate)
        metadata = metadata_score(query_metadata, candidate.get("metadata", {}))
        temporal = temporal_score(query_dates, candidate.get("metadata", {}).get("dates", []))
        breakdown = {
            "semantic": semantic_scores[index],
            "bm25": bm25_scores[index],
            "keyword": keyword,
            "metadata": metadata,
            "temporal": temporal,
        }
        final_score = sum(weights[field] * breakdown[field] for field in breakdown)
        ranked.append(
            {
                "chunk_id": candidate["chunk_id"],
                "document_id": candidate.get("document_id"),
                "chunk_index": candidate.get("chunk_index"),
                "text": candidate.get("text", ""),
                "section": candidate.get("section"),
                "final_score": _finite_clamped(final_score),
                "score_breakdown": breakdown,
                "retrieved_by": candidate.get("retrieved_by", []),
                "metadata": candidate.get("metadata", {}),
                "bm25_score": candidate.get("bm25_score"),
                "semantic_score": candidate.get("semantic_score"),
            }
        )

    ranked.sort(key=lambda result: (-result["final_score"], result["chunk_id"]))
    return ranked[:limit]


def ranking_weights() -> dict[str, float]:
    settings = get_settings()
    configured = {
        "semantic": settings.RANK_SEMANTIC_WEIGHT,
        "bm25": settings.RANK_BM25_WEIGHT,
        "keyword": settings.RANK_KEYWORD_WEIGHT,
        "metadata": settings.RANK_METADATA_WEIGHT,
        "temporal": settings.RANK_TEMPORAL_WEIGHT,
    }
    for name, value in configured.items():
        if not _is_finite_number(value):
            raise ValueError(f"Ranking weight '{name}' must be numeric and finite")
        if value < 0:
            raise ValueError(f"Ranking weight '{name}' must be non-negative")

    total = sum(configured.values())
    if total <= 0:
        raise ValueError("At least one ranking weight must be positive")
    return {name: value / total for name, value in configured.items()}


def normalize_scores(values: list[Any]) -> list[float]:
    """Min-max normalize finite scores while treating missing/invalid values as 0.0.

    A single meaningful candidate, or a candidate set where all meaningful scores
    are equal and positive, receives 1.0 for those meaningful scores. Equal zero,
    missing, NaN, and infinite values normalize to 0.0.
    """
    sanitized = [_safe_float(value) for value in values]
    meaningful = [value for value in sanitized if value is not None]
    if not meaningful:
        return [0.0 for _ in values]

    minimum = min(meaningful)
    maximum = max(meaningful)
    if maximum == minimum:
        normalized_value = 1.0 if maximum > 0 else 0.0
        return [normalized_value if value is not None else 0.0 for value in sanitized]

    return [
        0.0 if value is None else _finite_clamped((value - minimum) / (maximum - minimum))
        for value in sanitized
    ]


def meaningful_query_terms(query: str) -> list[str]:
    terms = []
    seen = set()
    for token in tokenize(query):
        if token in STOP_WORDS:
            continue
        if len(token) < 2 and not any(character.isdigit() for character in token):
            continue
        if token not in seen:
            seen.add(token)
            terms.append(token)
    return terms


def keyword_score(query_terms: list[str], candidate: dict) -> float:
    if not query_terms:
        return 0.0
    haystack = " ".join(
        [
            candidate.get("text", ""),
            candidate.get("section") or "",
            _metadata_text(candidate.get("metadata", {})),
        ]
    ).lower()
    haystack_tokens = set(tokenize(haystack))
    matches = 0
    for term in query_terms:
        if term in haystack_tokens or term in haystack:
            matches += 1
    return _finite_clamped(matches / len(query_terms))


def metadata_score(query_metadata: dict[str, list[str]], candidate_metadata: dict[str, Any]) -> float:
    expected = []
    matched = 0
    for field in METADATA_FIELDS:
        query_values = _lower_values(query_metadata.get(field, []))
        if not query_values:
            continue
        expected.extend(query_values)
        candidate_values = set(_lower_values(candidate_metadata.get(field, [])))
        matched += sum(1 for value in query_values if value in candidate_values)

    if not expected:
        return 0.0
    return _finite_clamped(matched / len(expected))


def temporal_score(query_dates: list[date], candidate_dates: list[str]) -> float:
    if not query_dates or not candidate_dates:
        return 0.0
    candidate_date_values = _date_values(candidate_dates)
    if not candidate_date_values:
        return 0.0

    decay_days = get_settings().RANK_TEMPORAL_DECAY_DAYS
    if not _is_finite_number(decay_days) or decay_days <= 0:
        raise ValueError("RANK_TEMPORAL_DECAY_DAYS must be positive")

    closest_days = min(abs((query_date - candidate_date).days) for query_date in query_dates for candidate_date in candidate_date_values)
    return _finite_clamped(math.exp(-closest_days / decay_days))


def _deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for candidate in candidates:
        chunk_id = candidate.get("chunk_id")
        if not chunk_id:
            continue
        existing = merged.setdefault(chunk_id, dict(candidate))
        if existing is candidate:
            continue
        existing["bm25_score"] = _prefer_score(existing.get("bm25_score"), candidate.get("bm25_score"))
        existing["semantic_score"] = _prefer_score(existing.get("semantic_score"), candidate.get("semantic_score"))
        existing["text"] = existing.get("text") or candidate.get("text", "")
        existing["section"] = existing.get("section") or candidate.get("section")
        existing["document_id"] = existing.get("document_id") or candidate.get("document_id")
        existing["chunk_index"] = existing.get("chunk_index") if existing.get("chunk_index") is not None else candidate.get("chunk_index")
        existing["metadata"] = _merge_metadata(existing.get("metadata", {}), candidate.get("metadata", {}))
        existing["retrieved_by"] = _merge_provenance(existing.get("retrieved_by", []), candidate.get("retrieved_by", []))
    return [merged[chunk_id] for chunk_id in sorted(merged)]


def _query_metadata(query: str) -> dict[str, list[str]]:
    extracted = extract_metadata("", query)
    metadata = extracted.model_dump()
    metadata["document_type"] = []
    return metadata


def _metadata_text(metadata: dict[str, Any]) -> str:
    values = []
    for value in metadata.values():
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value:
            values.append(str(value))
    return " ".join(values)


def _merge_metadata(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    merged = dict(first or {})
    for key, value in (second or {}).items():
        if key not in merged or not merged[key]:
            merged[key] = value
        elif isinstance(merged[key], list):
            merged[key] = _unique_preserve_order([*merged[key], *(_as_list(value))])
    return merged


def _merge_provenance(first: list[str], second: list[str]) -> list[str]:
    order = {"bm25": 0, "vector": 1}
    return sorted(_unique_preserve_order([*first, *second]), key=lambda value: order.get(value, 99))


def _prefer_score(first: Any, second: Any) -> Optional[float]:
    first_value = _safe_float(first)
    second_value = _safe_float(second)
    if first_value is None:
        return second_value
    if second_value is None:
        return first_value
    return max(first_value, second_value)


def _date_values(values: list[str]) -> list[date]:
    parsed = []
    for value in values:
        try:
            parsed.append(date.fromisoformat(str(value)))
        except ValueError:
            continue
    return parsed


def _lower_values(value: Any) -> list[str]:
    return [str(item).lower() for item in _as_list(value) if str(item).strip()]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _unique_preserve_order(values: list[Any]) -> list[Any]:
    seen = set()
    unique = []
    for value in values:
        key = str(value).lower()
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _finite_clamped(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
