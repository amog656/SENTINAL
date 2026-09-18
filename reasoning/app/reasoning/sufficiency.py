from __future__ import annotations

from app.models.schemas import InvestigationState


def assess_sufficiency(state: InvestigationState) -> tuple[str, list[str]]:
    critical = [goal for goal in state.goals if goal.critical_to_question]
    incomplete = [goal for goal in critical if goal.status != "supported"]
    if not incomplete:
        return "evidence_supported", []
    reasons = [f"{goal.description}: {goal.status}." for goal in incomplete]
    if state.loop_termination_reason == "max_hops_reached" and any(goal.status in {"not_investigated", "unresolved"} for goal in incomplete):
        return "unresolved", reasons
    # Reaching this branch means the planned search budget was not exhausted; the
    # engine looked for an answer but could not target a useful additional query.
    return "insufficient_evidence", reasons
