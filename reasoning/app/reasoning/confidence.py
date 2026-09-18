from __future__ import annotations

from app import config
from app.models.schemas import InvestigationState


def calculate_confidence(state: InvestigationState) -> tuple[float, dict[str, object]]:
    supporting = len(state.evidence)
    document_types = len({e.document_type for e in state.evidence})
    cross_confirmation = supporting >= 2 and document_types >= 2
    temporal_consistency = any(hypothesis.get("relationship_type") == "temporal_precedence" for hypothesis in state.hypotheses)
    entity_consistency = any(
        first.facts.get("service") and first.facts.get("service") == second.facts.get("service")
        for index, first in enumerate(state.evidence) for second in state.evidence[index + 1:]
    )
    missing = sum(goal.critical_to_question and goal.status != "supported" for goal in state.goals)
    # A conflict that was resolved (or was contextually compatible) is not a
    # confidence penalty. Do not double-count it as a generic evidence gap.
    unresolved = sum(item.get("resolution_status") == "unresolved" for item in state.resolved_conflicts)
    unsupported_assumptions = sum(hypothesis.get("relationship_type") == "causal_claim_without_evidence" for hypothesis in state.hypotheses)
    score = (
        config.CONFIDENCE_BASE
        + config.CONFIDENCE_PER_SUPPORTING_DOCUMENT * min(supporting, 5)
        + config.CONFIDENCE_PER_DOCUMENT_TYPE * document_types
        + (config.CONFIDENCE_CROSS_DOCUMENT_CONFIRMATION if cross_confirmation else 0)
        + (config.CONFIDENCE_TEMPORAL_CONSISTENCY_BONUS if temporal_consistency else 0)
        + (config.CONFIDENCE_ENTITY_CONSISTENCY_BONUS if entity_consistency else 0)
        - config.CONFIDENCE_UNRESOLVED_CONTRADICTION_PENALTY * max(unresolved, 0)
        - config.CONFIDENCE_MISSING_CRITICAL_EVIDENCE_PENALTY * missing
        - config.CONFIDENCE_UNSUPPORTED_ASSUMPTION_PENALTY * unsupported_assumptions
    )
    factors = {
        "formula": "base + supporting_documents + document_types + cross_document_confirmation - unresolved_contradictions - missing_critical_evidence",
        "supporting_documents": supporting, "document_types": document_types,
        "cross_document_confirmation": cross_confirmation, "contradictions_unresolved": max(unresolved, 0),
        "temporal_consistency": temporal_consistency, "entity_consistency": entity_consistency,
        "missing_critical_evidence": missing, "unsupported_assumptions": unsupported_assumptions,
    }
    return round(max(0.0, min(1.0, score)), 2), factors
