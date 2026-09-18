from functools import lru_cache
from typing import List

from app.config import get_settings


class EmbeddingError(Exception):
    pass


@lru_cache(maxsize=1)
def get_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError("sentence-transformers is not installed") from exc

    settings = get_settings()
    return SentenceTransformer(settings.EMBEDDING_MODEL)


def encode_text(text: str) -> List[float]:
    return encode_texts([text])[0]


def encode_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if any(not text or not text.strip() for text in texts):
        raise ValueError("Cannot encode empty text")

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    normalized = []
    for embedding in embeddings:
        if hasattr(embedding, "astype"):
            normalized.append(embedding.astype(float).tolist())
        else:
            normalized.append([float(value) for value in embedding])
    return normalized
