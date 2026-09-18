from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.models.schemas import Document
from app.retrieval.interface import RetrievalClient


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", value.lower()))


class MockRetrievalClient(RetrievalClient):
    """Small local fake; it is deliberately not a second vector database."""

    def __init__(self, documents_path: Path):
        self.documents = [Document(**item) for item in __import__("json").loads(documents_path.read_text())]

    async def search(self, query: str, filters: dict[str, Any] | None = None) -> list[Document]:
        filters = filters or {}
        query_tokens = _tokens(query)
        candidates: list[tuple[int, Document]] = []
        for document in self.documents:
            if any(getattr(document, key, None) != value for key, value in filters.items() if key != "limit"):
                continue
            haystack = " ".join([
                document.document_id, document.title, document.service or "", document.version or "",
                " ".join(document.claims), document.content,
            ])
            score = len(query_tokens & _tokens(haystack))
            if score:
                candidates.append((score, document))
        # For equal lexical relevance, favor the most recent evidence. This is a
        # transparent local-dev approximation; RETRIEVAL will own production ranking.
        candidates.sort(key=lambda item: (item[0], item[1].date.isoformat(), item[1].document_id), reverse=True)
        return [document for _, document in candidates[: int(filters.get("limit", 5))]]
