from __future__ import annotations

from app.models.schemas import Evidence


def extract_clues(evidence: Evidence) -> list[str]:
    clues = [str(value) for key, value in evidence.facts.items() if key in {"document_id", "document_type", "service", "version", "symptom", "date", "root_cause"}]
    for event in evidence.events:
        clues.extend(str(event[key]) for key in ("event", "timestamp", "date") if event.get(key))
    return list(dict.fromkeys(clues))
