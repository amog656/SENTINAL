from __future__ import annotations

import json
import re

from app.config import MAX_HOPS
from app.models.schemas import InvestigationGoal, InvestigationPlan


def create_plan(question: str) -> InvestigationPlan:
    lowered = question.lower()
    if any(term in lowered for term in ("restart", "troubleshooting", "guidance", "dependency failure")):
        service = "service-a" if "service a" in lowered or "service-a" in lowered else "service"
        return InvestigationPlan(
            initial_query=f"{service} latency troubleshooting",
            goals=[
                InvestigationGoal(id="guidance_evidence", description="Find applicable troubleshooting guidance"),
                InvestigationGoal(id="guidance_conflict", description="Detect and resolve conflicting troubleshooting guidance"),
                InvestigationGoal(id="evidence_gaps", description="Identify material evidence gaps", critical_to_question=False),
            ],
            max_hops=MAX_HOPS,
        )
    service = "orders-api" if "orders" in lowered and "api" in lowered else "service"
    symptom = "latency" if any(word in lowered for word in ("slow", "latency", "timeout")) else "incident"
    return InvestigationPlan(
        initial_query=f"{service} {symptom} incident",
        goals=[
            InvestigationGoal(id="current_incident", description="Identify the current incident"),
            InvestigationGoal(id="deployment_relationship", description="Determine whether the deployment is temporally related"),
            InvestigationGoal(id="historical_incident", description="Investigate whether a previous similar incident occurred"),
            InvestigationGoal(id="historical_comparison", description="Compare current and historical evidence"),
            InvestigationGoal(id="evidence_gaps", description="Identify material evidence gaps", critical_to_question=False),
        ],
        max_hops=MAX_HOPS,
    )


def deployment_query(service: str | None, version: str | None) -> str:
    return " ".join(part for part in [service or "service", version or "", "deployment"] if part)


def historical_query(service: str | None, symptom: str | None) -> str:
    return " ".join(part for part in [service or "service", symptom or "incident", "postmortem"] if part)


def next_query_fallback(goal_id: str, service: str | None, version: str | None, symptom: str | None) -> str | None:
    """Deterministic local fallback for the cache-backed follow-up proposal."""
    if goal_id == "deployment_relationship":
        return deployment_query(service, version)
    if goal_id == "historical_incident":
        return historical_query(service, symptom)
    if goal_id == "guidance_conflict":
        return " ".join(part for part in [service or "service", "dependency", "troubleshooting"] if part)
    return None


def encode_follow_up(goal_id: str, query: str | None) -> str:
    return json.dumps({"goal_id": goal_id, "query": query})


def decode_follow_up(value: str) -> tuple[str | None, str | None]:
    try:
        data = json.loads(value)
        return data.get("goal_id"), data.get("query")
    except (TypeError, ValueError):
        return None, None


def requested_date(question: str) -> str | None:
    match = re.search(r"20\d{2}-\d{2}-\d{2}", question)
    return match.group(0) if match else None
