from __future__ import annotations

from app.models.schemas import InvestigationState


def generate_report(state: InvestigationState) -> dict:
    evidence_ids = [item.document_id for item in state.evidence]
    known = []
    not_established = []
    if state.incidents:
        known.append(f"Current incident evidence: {', '.join(state.incidents)}.")
    if state.deployments:
        known.append(f"Deployment evidence: {', '.join(state.deployments)}.")
    comparison = state.historical_comparisons[0] if state.historical_comparisons else None
    if comparison and comparison["comparison_performed"]:
        known.append("A historical incident has a similar symptom.")
        not_established.append("The current and historical incidents have the same root cause.")
    if any(item.get("relationship_type") == "temporal_precedence" for item in state.hypotheses):
        known.append("The deployment temporally preceded the incident.")
        not_established.append("The deployment caused the incident.")
    if comparison and comparison["comparison_performed"] and comparison["same_root_cause"] == "not_established":
        conclusion = f"The current incident shares service and symptom with prior incident {comparison['historical_document_id']}, but the evidence does not establish the same root cause."
    elif state.status == "evidence_supported":
        conclusion = "Evidence supports the requested investigation goals, while causal claims remain limited to what the sources establish."
    elif state.status == "unresolved":
        conclusion = "The investigation reached its hop budget before all critical goals could be resolved."
    else:
        conclusion = "Insufficient evidence to establish every requested conclusion; the report preserves what is known and what remains unproven."
    return {
        "investigation_id": state.investigation_id, "question": state.original_question,
        "status": state.status, "conclusion": conclusion, "confidence": state.confidence,
        "confidence_factors": state.confidence_factors, "timeline": state.timeline,
        "key_evidence": [{"document_id": evidence.document_id, "facts": evidence.facts, "claims": evidence.claims} for evidence in state.evidence],
        "historical_comparisons": state.historical_comparisons, "contradictions": state.contradictions,
        "resolved_conflicts": state.resolved_conflicts, "open_questions": state.open_questions,
        "investigation_steps": [event.model_dump() for event in state.investigation_steps],
        "source_document_ids": evidence_ids,
        "goals": [goal.model_dump() for goal in state.goals],
        "loop_termination_reason": state.loop_termination_reason,
        "what_is_known": known,
        "what_is_not_established": list(dict.fromkeys(not_established)),
    }
