import os
from functools import lru_cache
from typing import Optional


class Settings:
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "investigator")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "investigator")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "investigation")

    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "investigation_chunks")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    MAX_SEARCH_LIMIT: int = int(os.getenv("MAX_SEARCH_LIMIT", "50"))
    HYBRID_BM25_WEIGHT: float = float(os.getenv("HYBRID_BM25_WEIGHT", "0.5"))
    HYBRID_SEMANTIC_WEIGHT: float = float(os.getenv("HYBRID_SEMANTIC_WEIGHT", "0.5"))
    RANK_SEMANTIC_WEIGHT: float = float(os.getenv("RANK_SEMANTIC_WEIGHT", "0.40"))
    RANK_BM25_WEIGHT: float = float(os.getenv("RANK_BM25_WEIGHT", "0.30"))
    RANK_KEYWORD_WEIGHT: float = float(os.getenv("RANK_KEYWORD_WEIGHT", "0.15"))
    RANK_METADATA_WEIGHT: float = float(os.getenv("RANK_METADATA_WEIGHT", "0.10"))
    RANK_TEMPORAL_WEIGHT: float = float(os.getenv("RANK_TEMPORAL_WEIGHT", "0.05"))
    RANK_TEMPORAL_DECAY_DAYS: float = float(os.getenv("RANK_TEMPORAL_DECAY_DAYS", "30"))
    INVESTIGATION_MAX_HOPS: int = int(os.getenv("INVESTIGATION_MAX_HOPS", "3"))
    INVESTIGATION_MAX_CLUES_PER_HOP: int = int(os.getenv("INVESTIGATION_MAX_CLUES_PER_HOP", "5"))
    INVESTIGATION_RESULTS_PER_HOP: int = int(os.getenv("INVESTIGATION_RESULTS_PER_HOP", "5"))
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1600"))
    AGENT_MAX_EVIDENCE: int = int(os.getenv("AGENT_MAX_EVIDENCE", "20"))

    @property
    def postgres_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def qdrant_url(self) -> str:
        if self.QDRANT_URL:
            return self.QDRANT_URL
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
