import pytest

from app.embeddings import encoder


class FakeModel:
    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        return [[len(text), sum(ord(char) for char in text) % 100] for text in texts]


def test_encoder_returns_vectors(monkeypatch):
    monkeypatch.setattr(encoder, "get_embedding_model", lambda: FakeModel())

    vector = encoder.encode_text("orders-api latency")

    assert isinstance(vector, list)
    assert len(vector) == 2
    assert all(isinstance(value, float) for value in vector)


def test_same_text_produces_deterministic_embeddings(monkeypatch):
    monkeypatch.setattr(encoder, "get_embedding_model", lambda: FakeModel())

    assert encoder.encode_text("same text") == encoder.encode_text("same text")


def test_batch_encoding(monkeypatch):
    monkeypatch.setattr(encoder, "get_embedding_model", lambda: FakeModel())

    vectors = encoder.encode_texts(["alpha", "beta"])

    assert len(vectors) == 2
    assert vectors[0] != vectors[1]


def test_empty_text_is_rejected(monkeypatch):
    monkeypatch.setattr(encoder, "get_embedding_model", lambda: FakeModel())

    with pytest.raises(ValueError, match="empty text"):
        encoder.encode_text("")
