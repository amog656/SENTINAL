from typing import Any

from app.ingestion.metadata import extract_dates, extract_metadata
from app.investigation.models import Clue


CLUE_PRIORITY = {
    "deployment_id": 0,
    "incident_id": 1,
    "version": 2,
    "service": 3,
    "technical_entity": 4,
    "date": 5,
}


def extract_clues_from_evidence(evidence_items: list[dict[str, Any] | Any]) -> list[Clue]:
    clues: list[Clue] = []
    seen = set()
    for item in evidence_items:
        data = _as_dict(item)
        text = data.get("text", "")
        metadata = data.get("metadata", {}) or {}
        combined_text = " ".join([text, _metadata_text(metadata)])
        extracted = extract_metadata("", combined_text).model_dump()
        extracted["dates"] = extract_dates(combined_text)

        for clue_type, field in (
            ("service", "services"),
            ("version", "versions"),
            ("incident_id", "incident_ids"),
            ("deployment_id", "deployment_ids"),
            ("date", "dates"),
            ("technical_entity", "technical_entities"),
        ):
            for raw_value in extracted.get(field, []):
                value = normalize_clue_value(raw_value, clue_type)
                key = (clue_type, value.lower(), data.get("chunk_id"))
                if not value or key in seen:
                    continue
                seen.add(key)
                clues.append(
                    Clue(
                        value=value,
                        type=clue_type,
                        source_chunk_id=data.get("chunk_id", ""),
                        source_document_id=data.get("document_id"),
                        hop=int(data.get("hop", 0)),
                    )
                )

    clues.sort(key=lambda clue: (CLUE_PRIORITY.get(clue.type, 99), clue.value.lower(), clue.source_chunk_id))
    return clues


def normalize_clue_value(value: str, clue_type: str) -> str:
    value = str(value).strip()
    if clue_type in {"incident_id", "deployment_id"}:
        return value.upper()
    if clue_type in {"service", "version", "technical_entity"}:
        return value.lower()
    return value


def clue_key(clue: Clue) -> tuple[str, str]:
    return (clue.type, clue.value.lower())


def clue_query(clue: Clue) -> str:
    return clue.value


def _as_dict(item: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return dict(item)


def _metadata_text(metadata: dict[str, Any]) -> str:
    values = []
    for value in metadata.values():
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value:
            values.append(str(value))
    return " ".join(values)
