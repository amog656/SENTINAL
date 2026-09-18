from collections.abc import Callable
from typing import Any, Optional

from app.config import get_settings
from app.investigation.clues import clue_key, clue_query, extract_clues_from_evidence
from app.investigation.models import (
    Clue,
    EvidenceItem,
    EvidenceRelationship,
    InvestigationHop,
    InvestigationResult,
)
from app.investigation.trace import InvestigationTrace


RankedSearch = Callable[[str, int], list[dict[str, Any]]]


def run_investigation(
    query: str,
    ranked_searcher: RankedSearch,
    max_hops: Optional[int] = None,
    results_per_hop: Optional[int] = None,
    max_clues_per_hop: Optional[int] = None,
) -> InvestigationResult:
    settings = get_settings()
    max_hops = settings.INVESTIGATION_MAX_HOPS if max_hops is None else max_hops
    results_per_hop = settings.INVESTIGATION_RESULTS_PER_HOP if results_per_hop is None else results_per_hop
    max_clues_per_hop = settings.INVESTIGATION_MAX_CLUES_PER_HOP if max_clues_per_hop is None else max_clues_per_hop
    _validate_request(query, max_hops, results_per_hop, max_clues_per_hop)

    trace = InvestigationTrace()
    trace.record("investigation_started", 0, query=query, max_hops=max_hops, results_per_hop=results_per_hop)

    evidence_by_chunk: dict[str, EvidenceItem] = {}
    all_clues_by_key: dict[tuple[str, str, str], Clue] = {}
    searched_clues: set[tuple[str, str]] = set()
    relationships: list[EvidenceRelationship] = []
    relationship_keys: set[tuple[str, str, str, Optional[str]]] = set()
    hops: list[InvestigationHop] = []

    initial_results = ranked_searcher(query, results_per_hop)
    initial_new = _add_results(
        evidence_by_chunk,
        relationships,
        relationship_keys,
        initial_results,
        hop=0,
        retrieval_query=query,
        trigger_clue=None,
        trace=trace,
    )
    hops.append(InvestigationHop(hop=0, query=query, results_found=len(initial_results), new_evidence_found=initial_new))
    trace.record("initial_search_completed", 0, query=query, results_found=len(initial_results), new_evidence_found=initial_new)

    current_evidence = [_with_context(result, 0, query, None) for result in initial_results]
    hops_completed = 0
    follow_up_searches = 0

    for hop in range(1, max_hops + 1):
        clues = extract_clues_from_evidence(current_evidence)
        _record_clues(all_clues_by_key, clues)
        trace.record("clues_extracted", hop - 1, clues_found=len(clues))

        new_clues = _unique_search_clues([clue for clue in clues if clue_key(clue) not in searched_clues])
        new_clues = new_clues[:max_clues_per_hop]
        if not new_clues:
            trace.record("investigation_stopped", hop - 1, reason="no_new_clues")
            break

        next_evidence = []
        hop_results_found = 0
        hop_new_evidence = 0
        for clue in new_clues:
            searched_clues.add(clue_key(clue))
            search_query = clue_query(clue)
            results = ranked_searcher(search_query, results_per_hop)
            follow_up_searches += 1
            hop_results_found += len(results)
            trace.record(
                "follow_up_search_executed",
                hop,
                query=search_query,
                trigger_clue=clue.model_dump(),
                results_found=len(results),
            )
            added = _add_results(
                evidence_by_chunk,
                relationships,
                relationship_keys,
                results,
                hop=hop,
                retrieval_query=search_query,
                trigger_clue=clue,
                trace=trace,
            )
            hop_new_evidence += added
            for result in results:
                next_evidence.append(_with_context(result, hop, search_query, clue))

        hops.append(
            InvestigationHop(
                hop=hop,
                query=", ".join(clue_query(clue) for clue in new_clues),
                trigger_clue=new_clues[0] if len(new_clues) == 1 else None,
                results_found=hop_results_found,
                new_evidence_found=hop_new_evidence,
            )
        )
        hops_completed = hop
        trace.record("hop_completed", hop, results_found=hop_results_found, new_evidence_found=hop_new_evidence)

        if not next_evidence or hop_new_evidence == 0:
            trace.record("investigation_stopped", hop, reason="no_new_evidence")
            break
        current_evidence = next_evidence

    evidence = sorted(evidence_by_chunk.values(), key=lambda item: (-item.final_score, item.chunk_id))
    clues = sorted(all_clues_by_key.values(), key=lambda clue: (clue.hop, clue.type, clue.value.lower(), clue.source_chunk_id))
    relationships = sorted(relationships, key=lambda item: (item.hop, item.from_chunk_id, item.to_chunk_id, item.relationship, item.clue or ""))
    trace.record("investigation_completed", hops_completed, evidence_count=len(evidence), clue_count=len(clues), relationship_count=len(relationships))

    return InvestigationResult(
        query=query,
        max_hops=max_hops,
        hops_completed=hops_completed,
        evidence=evidence,
        clues=clues,
        relationships=relationships,
        hops=hops,
        trace=trace.as_list(),
        summary={
            "documents_examined": len({item.document_id for item in evidence if item.document_id}),
            "chunks_examined": len(evidence),
            "hops_completed": hops_completed,
            "unique_clues_found": len({clue_key(clue) for clue in clues}),
            "follow_up_searches": follow_up_searches,
        },
    )


def _validate_request(query: str, max_hops: int, results_per_hop: int, max_clues_per_hop: int) -> None:
    settings = get_settings()
    if not query or not query.strip():
        raise ValueError("query cannot be empty")
    if max_hops < 1 or max_hops > settings.INVESTIGATION_MAX_HOPS:
        raise ValueError(f"max_hops must be between 1 and {settings.INVESTIGATION_MAX_HOPS}")
    if results_per_hop < 1 or results_per_hop > settings.MAX_SEARCH_LIMIT:
        raise ValueError(f"results_per_hop must be between 1 and {settings.MAX_SEARCH_LIMIT}")
    if max_clues_per_hop < 1:
        raise ValueError("max_clues_per_hop must be positive")


def _add_results(
    evidence_by_chunk: dict[str, EvidenceItem],
    relationships: list[EvidenceRelationship],
    relationship_keys: set[tuple[str, str, str, Optional[str]]],
    results: list[dict[str, Any]],
    hop: int,
    retrieval_query: str,
    trigger_clue: Optional[Clue],
    trace: InvestigationTrace,
) -> int:
    new_count = 0
    existing_items = list(evidence_by_chunk.values())
    for result in results:
        item = _evidence_item(result, hop, retrieval_query, trigger_clue)
        existing = evidence_by_chunk.get(item.chunk_id)
        if existing:
            _merge_evidence(existing, item)
            trace.record("duplicate_evidence_skipped", hop, chunk_id=item.chunk_id, query=retrieval_query)
        else:
            evidence_by_chunk[item.chunk_id] = item
            existing_items.append(item)
            new_count += 1
            trace.record("evidence_discovered", hop, chunk_id=item.chunk_id, query=retrieval_query)
        if trigger_clue and trigger_clue.source_chunk_id != item.chunk_id:
            _add_relationship(
                relationships,
                relationship_keys,
                trigger_clue.source_chunk_id,
                item.chunk_id,
                "triggered_by_clue",
                trigger_clue.value,
                hop,
            )
        _add_shared_relationships(relationships, relationship_keys, existing_items, item, hop)
    return new_count


def _evidence_item(result: dict[str, Any], hop: int, retrieval_query: str, trigger_clue: Optional[Clue]) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=result["chunk_id"],
        document_id=result.get("document_id"),
        chunk_index=result.get("chunk_index"),
        text=result.get("text", ""),
        section=result.get("section"),
        final_score=float(result.get("final_score") or 0.0),
        score_breakdown=result.get("score_breakdown", {}),
        retrieved_by=result.get("retrieved_by", []),
        metadata=result.get("metadata", {}),
        bm25_score=result.get("bm25_score"),
        semantic_score=result.get("semantic_score"),
        hop=hop,
        retrieval_query=retrieval_query,
        trigger_clue=trigger_clue,
        discovered_at_hops=[hop],
        discovery_queries=[retrieval_query],
        discovered_via_clues=[trigger_clue] if trigger_clue else [],
    )


def _merge_evidence(existing: EvidenceItem, new_item: EvidenceItem) -> None:
    if new_item.final_score > existing.final_score:
        existing.final_score = new_item.final_score
        existing.score_breakdown = new_item.score_breakdown
        existing.bm25_score = new_item.bm25_score
        existing.semantic_score = new_item.semantic_score
        existing.retrieved_by = _unique([*existing.retrieved_by, *new_item.retrieved_by])
    existing.discovered_at_hops = sorted(set([*existing.discovered_at_hops, *new_item.discovered_at_hops]))
    existing.discovery_queries = _unique([*existing.discovery_queries, *new_item.discovery_queries])
    existing.discovered_via_clues = _unique_clues([*existing.discovered_via_clues, *new_item.discovered_via_clues])


def _record_clues(all_clues_by_key: dict[tuple[str, str, str], Clue], clues: list[Clue]) -> None:
    for clue in clues:
        all_clues_by_key[(clue.type, clue.value.lower(), clue.source_chunk_id)] = clue


def _with_context(result: dict[str, Any], hop: int, retrieval_query: str, trigger_clue: Optional[Clue]) -> dict[str, Any]:
    data = dict(result)
    data["hop"] = hop
    data["retrieval_query"] = retrieval_query
    data["trigger_clue"] = trigger_clue.model_dump() if trigger_clue else None
    return data


def _add_shared_relationships(
    relationships: list[EvidenceRelationship],
    relationship_keys: set[tuple[str, str, str, Optional[str]]],
    existing_items: list[EvidenceItem],
    item: EvidenceItem,
    hop: int,
) -> None:
    for other in existing_items:
        if other.chunk_id == item.chunk_id:
            continue
        for relationship, field in (
            ("same_incident", "incident_ids"),
            ("same_deployment", "deployment_ids"),
            ("same_service", "services"),
            ("same_version", "versions"),
            ("same_technical_entity", "technical_entities"),
        ):
            shared = sorted(set(_lower_values(other.metadata.get(field, []))).intersection(_lower_values(item.metadata.get(field, []))))
            for value in shared:
                _add_relationship(relationships, relationship_keys, other.chunk_id, item.chunk_id, relationship, value, hop)


def _add_relationship(
    relationships: list[EvidenceRelationship],
    relationship_keys: set[tuple[str, str, str, Optional[str]]],
    from_chunk_id: str,
    to_chunk_id: str,
    relationship: str,
    clue: Optional[str],
    hop: int,
) -> None:
    key = (from_chunk_id, to_chunk_id, relationship, clue)
    if key in relationship_keys:
        return
    relationship_keys.add(key)
    relationships.append(
        EvidenceRelationship(
            from_chunk_id=from_chunk_id,
            to_chunk_id=to_chunk_id,
            relationship=relationship,
            clue=clue,
            hop=hop,
        )
    )


def _lower_values(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [str(item).lower() for item in values if str(item).strip()]


def _unique(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _unique_clues(clues: list[Clue]) -> list[Clue]:
    seen = set()
    unique = []
    for clue in clues:
        key = (clue.type, clue.value.lower(), clue.source_chunk_id)
        if key not in seen:
            seen.add(key)
            unique.append(clue)
    return unique


def _unique_search_clues(clues: list[Clue]) -> list[Clue]:
    seen = set()
    unique = []
    for clue in clues:
        key = clue_key(clue)
        if key not in seen:
            seen.add(key)
            unique.append(clue)
    return unique
