from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.models.schemas import Document


class RetrievalClient(ABC):
    """Replace this implementation when RETRIEVAL exposes its HTTP/API contract."""

    @abstractmethod
    async def search(self, query: str, filters: dict[str, Any] | None = None) -> list[Document]:
        """Return only documents conforming to the frozen Document schema."""
        raise NotImplementedError
