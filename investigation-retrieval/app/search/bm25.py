import math
import re
from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.config import get_settings
from app.storage.documents import list_documents_with_chunks
from app.storage.postgres import get_db_context

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - used only when the dependency is unavailable locally
    BM25Okapi = None


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
FILTER_FIELDS = {"services", "document_type", "versions", "incident_ids", "deployment_ids"}


@dataclass(frozen=True)
class BM25Chunk:
    chunk_id: str
    document_id: str
    text: str
    section: Optional[str]
    metadata: Dict[str, Any]


class SimpleBM25Okapi:
    """Small fallback with the same get_scores shape used from rank-bm25."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        self.doc_len = [len(document) for document in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self._initialize()

    def _initialize(self) -> None:
        document_frequency: dict[str, int] = {}
        for document in self.corpus:
            frequencies: dict[str, int] = {}
            for token in document:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.doc_freqs.append(frequencies)
            for token in frequencies:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        corpus_size = len(self.corpus)
        for token, frequency in document_frequency.items():
            self.idf[token] = math.log(1 + (corpus_size - frequency + 0.5) / (frequency + 0.5))

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for index, frequencies in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_len[index]
            for token in query_tokens:
                token_frequency = frequencies.get(token, 0)
                if not token_frequency:
                    continue
                denominator = token_frequency + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
                score += self.idf.get(token, 0.0) * (token_frequency * (self.k1 + 1)) / denominator
            scores.append(score)
        return scores


class BM25Index:
    def __init__(self, chunks: list[BM25Chunk]):
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(_searchable_text(chunk)) for chunk in chunks]
        model_class = BM25Okapi or SimpleBM25Okapi
        self.model = model_class(self.tokenized_corpus) if self.tokenized_corpus else None

    def search(self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> list[dict]:
        validate_search_input(query, limit)
        if not self.model:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            raise ValueError("query cannot be empty")

        scores = list(self.model.get_scores(query_tokens))
        candidates = []
        for chunk, score in zip(self.chunks, scores):
            score = float(score)
            if score <= 0:
                continue
            if not _matches_filters(chunk.metadata, filters):
                continue
            candidates.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "score": score,
                    "text": chunk.text,
                    "section": chunk.section,
                    "metadata": chunk.metadata,
                }
            )

        candidates.sort(key=lambda result: (-result["score"], result["chunk_id"]))
        return candidates[:limit]


_bm25_index: Optional[BM25Index] = None
_index_lock = RLock()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def validate_search_input(query: str, limit: int) -> None:
    max_limit = get_settings().MAX_SEARCH_LIMIT
    if not query or not query.strip():
        raise ValueError("query cannot be empty")
    if limit < 1 or limit > max_limit:
        raise ValueError(f"limit must be between 1 and {max_limit}")


def build_bm25_index(db: Session) -> BM25Index:
    chunks = []
    for document in list_documents_with_chunks(db):
        metadata = _document_metadata(document)
        for chunk in sorted(document.chunks, key=lambda item: item.chunk_index):
            chunks.append(
                BM25Chunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    section=chunk.section,
                    metadata=metadata,
                )
            )
    return BM25Index(chunks)


def rebuild_bm25_index(db: Session) -> dict:
    global _bm25_index
    index = build_bm25_index(db)
    with _index_lock:
        _bm25_index = index
    return {"indexed_chunks": len(index.chunks)}


def search_bm25(query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None, db: Optional[Session] = None) -> list[dict]:
    validate_search_input(query, limit)
    index = _get_or_build_index(db)
    return index.search(query, limit=limit, filters=filters)


def _get_or_build_index(db: Optional[Session] = None) -> BM25Index:
    global _bm25_index
    with _index_lock:
        if _bm25_index is not None:
            return _bm25_index

    if db is not None:
        rebuild_bm25_index(db)
    else:
        with get_db_context() as context_db:
            rebuild_bm25_index(context_db)

    with _index_lock:
        return _bm25_index or BM25Index([])


def _searchable_text(chunk: BM25Chunk) -> str:
    metadata_terms = []
    for field in ("services", "versions", "incident_ids", "deployment_ids", "technical_entities"):
        value = chunk.metadata.get(field, [])
        metadata_terms.extend(value if isinstance(value, list) else [value])
    return " ".join([chunk.text, *[str(term) for term in metadata_terms if term]])


def _document_metadata(document: Any) -> Dict[str, Any]:
    return {
        "document_type": document.document_type,
        "services": document.services or [],
        "versions": document.versions or [],
        "dates": document.dates or [],
        "incident_ids": document.incident_ids or [],
        "deployment_ids": document.deployment_ids or [],
        "technical_entities": document.technical_entities or [],
    }


def _matches_filters(metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    if not filters:
        return True
    for field, expected in filters.items():
        if field not in FILTER_FIELDS:
            raise ValueError(f"Unsupported filter field: {field}")
        expected_values = _as_lower_list(expected)
        actual = metadata.get(field)
        if field == "document_type":
            if str(actual or "").lower() not in expected_values:
                return False
            continue
        actual_values = _as_lower_list(actual or [])
        if not set(expected_values).intersection(actual_values):
            return False
    return True


def _as_lower_list(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [str(item).lower() for item in values if str(item).strip()]
