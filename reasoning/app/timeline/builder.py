from __future__ import annotations

from app.models.schemas import Evidence


def build_timeline(evidence: list[Evidence]) -> list[dict]:
    """Sort real timestamps where known; date-only evidence stays date-only."""
    events = [dict(event) for item in evidence for event in item.events]
    return sorted(events, key=lambda event: (event["timestamp"] or event["date"], event["source"]))
