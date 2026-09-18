from app.investigation.investigator import run_investigation


def result(chunk_id, text, final_score=0.8, metadata=None):
    return {
        "chunk_id": chunk_id,
        "document_id": chunk_id.split("-C")[0] if "-C" in chunk_id else "DOC",
        "chunk_index": 1,
        "text": text,
        "section": "Root Cause",
        "final_score": final_score,
        "score_breakdown": {"semantic": final_score},
        "retrieved_by": ["bm25"],
        "metadata": metadata or {},
        "bm25_score": 5.0,
        "semantic_score": 0.8,
    }


def test_multi_hop_investigation_follows_explicit_clues():
    calls = []
    responses = {
        "Why did orders-api become slow after DEP-882?": [
            result("INC-1042-C001", "INC-1042 mentions DEP-882 for orders-api.", metadata={"deployment_ids": ["DEP-882"]})
        ],
        "DEP-882": [
            result("DEP-882-C001", "Deployment DEP-882 shipped orders-api v2.8.1.", metadata={"versions": ["v2.8.1"], "deployment_ids": ["DEP-882"]})
        ],
        "v2.8.1": [
            result("PM-211-C004", "v2.8.1 postmortem mentions connection pool exhaustion.", metadata={"versions": ["v2.8.1"], "technical_entities": ["connection pool exhaustion"]})
        ],
    }

    def searcher(query, limit):
        calls.append(query)
        return responses.get(query, [])

    investigation = run_investigation(
        "Why did orders-api become slow after DEP-882?",
        searcher,
        max_hops=2,
        results_per_hop=1,
        max_clues_per_hop=1,
    )
    by_id = {item.chunk_id: item for item in investigation.evidence}

    assert calls == ["Why did orders-api become slow after DEP-882?", "DEP-882", "v2.8.1"]
    assert by_id["INC-1042-C001"].hop == 0
    assert by_id["DEP-882-C001"].hop == 1
    assert by_id["PM-211-C004"].hop == 2
    assert by_id["DEP-882-C001"].trigger_clue.value == "DEP-882"
    assert by_id["PM-211-C004"].trigger_clue.value == "v2.8.1"


def test_relationship_records_triggered_by_clue():
    responses = {
        "root query": [result("C1", "DEP-882")],
        "DEP-882": [result("C2", "deployment note")],
    }
    investigation = run_investigation("root query", lambda query, limit: responses.get(query, []), max_hops=1, results_per_hop=2)

    relationships = [relationship for relationship in investigation.relationships if relationship.relationship == "triggered_by_clue"]
    assert relationships[0].from_chunk_id == "C1"
    assert relationships[0].to_chunk_id == "C2"
    assert relationships[0].clue == "DEP-882"


def test_loop_prevention_searches_duplicate_clue_once():
    calls = []
    responses = {
        "root query": [result("C1", "DEP-882"), result("C2", "DEP-882")],
        "DEP-882": [result("C3", "DEP-882")],
    }

    def searcher(query, limit):
        calls.append(query)
        return responses.get(query, [])

    run_investigation("root query", searcher, max_hops=3, results_per_hop=5)

    assert calls.count("DEP-882") == 1


def test_max_hops_prevents_deeper_searches():
    calls = []
    responses = {
        "root query": [result("C1", "DEP-882")],
        "DEP-882": [result("C2", "v2.8.1")],
        "v2.8.1": [result("C3", "INC-1042")],
    }

    def searcher(query, limit):
        calls.append(query)
        return responses.get(query, [])

    investigation = run_investigation("root query", searcher, max_hops=1, results_per_hop=1, max_clues_per_hop=1)

    assert "v2.8.1" not in calls
    assert investigation.hops_completed == 1


def test_no_new_clues_terminates_early():
    investigation = run_investigation("root query", lambda query, limit: [result("C1", "ordinary text")], max_hops=3, results_per_hop=1)

    assert investigation.hops_completed == 0
    assert investigation.summary["follow_up_searches"] == 0


def test_duplicate_evidence_appears_once_with_discovery_history():
    responses = {
        "root query": [result("C1", "DEP-882", final_score=0.5)],
        "DEP-882": [result("C1", "DEP-882", final_score=0.9)],
    }
    investigation = run_investigation("root query", lambda query, limit: responses.get(query, []), max_hops=1, results_per_hop=2)

    assert len(investigation.evidence) == 1
    assert investigation.evidence[0].final_score == 0.9
    assert investigation.evidence[0].discovered_at_hops == [0, 1]
    assert investigation.evidence[0].discovery_queries == ["root query", "DEP-882"]


def test_investigation_is_deterministic():
    responses = {
        "root query": [result("C1", "DEP-882")],
        "DEP-882": [result("C2", "v2.8.1")],
    }
    first = run_investigation("root query", lambda query, limit: responses.get(query, []), max_hops=1, results_per_hop=2)
    second = run_investigation("root query", lambda query, limit: responses.get(query, []), max_hops=1, results_per_hop=2)

    assert first.model_dump() == second.model_dump()
