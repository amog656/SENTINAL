from __future__ import annotations

from app.models.schemas import Evidence


def compare_incidents(current: Evidence, historical: Evidence) -> dict:
    same_service = current.facts.get("service") == historical.facts.get("service") and current.facts.get("service") is not None
    same_symptom = current.facts.get("symptom") == historical.facts.get("symptom") and current.facts.get("symptom") is not None
    current_root_cause = current.facts.get("root_cause")
    historical_root_cause = historical.facts.get("root_cause")
    root_cause_established = bool(current_root_cause and historical_root_cause and current_root_cause == historical_root_cause)
    return {
        "comparison_performed": True,
        "current_document_id": current.document_id,
        "historical_document_id": historical.document_id,
        "source_document_ids": [current.document_id, historical.document_id],
        "similarity": "medium" if same_service and same_symptom else "low",
        # Keep the existing scalar `similarity` for EXPERIENCE compatibility;
        # this additive detail distinguishes the dimensions used to reach it.
        "similarity_dimensions": {
            "service": "high" if same_service else "low",
            "symptom": "high" if same_symptom else "low",
            "root_cause": "high" if root_cause_established else "unknown",
        },
        "same_service": same_service,
        "same_symptom": same_symptom,
        "current_version": current.facts.get("version"),
        "historical_version": historical.facts.get("version"),
        "historical_root_cause": historical_root_cause,
        "same_root_cause": "established" if root_cause_established else "not_established",
        "conclusion": "Similar symptom, but same root cause is not established." if same_service and same_symptom and not root_cause_established else "Similarity is limited by the available evidence.",
    }


def unperformable_comparison(current: Evidence, reason: str) -> dict:
    """Represent a comparison that could not be attempted, without pretending it ran."""
    return {
        "comparison_performed": False,
        "current_document_id": current.document_id,
        "historical_document_id": None,
        "source_document_ids": [current.document_id],
        "similarity": "unknown",
        "similarity_dimensions": {"service": "unknown", "symptom": "unknown", "root_cause": "unknown"},
        "same_service": False,
        "same_symptom": False,
        "current_version": current.facts.get("version"),
        "historical_version": None,
        "historical_root_cause": None,
        "same_root_cause": "not_established",
        "conclusion": reason,
    }
