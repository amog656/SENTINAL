from app.investigation.clues import clue_key, extract_clues_from_evidence


def evidence(text, metadata=None, chunk_id="C1"):
    return {
        "chunk_id": chunk_id,
        "document_id": "D1",
        "hop": 0,
        "text": text,
        "metadata": metadata or {},
    }


def test_extracts_explicit_clues_and_normalizes_values():
    clues = extract_clues_from_evidence(
        [
            evidence(
                "inc-1042 orders-api V2.8.1 slowed after dep-882 because of PostgreSQL connection pool exhaustion.",
                metadata={"technical_entities": ["connection pool exhaustion"]},
            )
        ]
    )
    by_type = {(clue.type, clue.value) for clue in clues}

    assert ("incident_id", "INC-1042") in by_type
    assert ("deployment_id", "DEP-882") in by_type
    assert ("service", "orders-api") in by_type
    assert ("version", "v2.8.1") in by_type
    assert ("technical_entity", "postgresql") in by_type
    assert ("technical_entity", "connection pool exhaustion") in by_type


def test_clue_extraction_does_not_hallucinate_absent_ids_or_versions():
    clues = extract_clues_from_evidence([evidence("orders-api became slow.")])
    values = {clue.value for clue in clues}

    assert "orders-api" in values
    assert "INC-1042" not in values
    assert "DEP-882" not in values
    assert "v2.8.1" not in values


def test_duplicate_clue_key_collapses_across_chunks_for_search_tracking():
    clues = extract_clues_from_evidence(
        [
            evidence("DEP-882 orders-api", chunk_id="C1"),
            evidence("dep-882 payment-service", chunk_id="C2"),
        ]
    )
    unique_search_clues = {clue_key(clue) for clue in clues}

    assert ("deployment_id", "dep-882") in unique_search_clues
    assert len([key for key in unique_search_clues if key == ("deployment_id", "dep-882")]) == 1
