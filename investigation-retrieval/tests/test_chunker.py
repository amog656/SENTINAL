from app.ingestion.chunker import chunk_document


def test_long_document_creates_multiple_chunks():
    text = "Summary\n" + ("orders-api v2.8.1 latency INC-1042 DEP-882. " * 200)
    chunks = chunk_document("INC-1042", text, {"services": ["orders-api"]}, target_chars=900, overlap_chars=120)

    assert len(chunks) > 1
    assert chunks[0].section == "Summary"


def test_chunks_overlap_appropriately():
    text = ("alpha beta gamma delta " * 180).strip()
    chunks = chunk_document("DOC-1", text, {}, target_chars=600, overlap_chars=100)

    assert chunks[0].text[-60:] in chunks[1].text


def test_chunk_ids_are_deterministic():
    text = "Timeline\n" + ("orders-api latency. " * 120)
    first = chunk_document("INC-1042", text, {}, target_chars=600, overlap_chars=100)
    second = chunk_document("INC-1042", text, {}, target_chars=600, overlap_chars=100)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
