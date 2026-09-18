from __future__ import annotations

import re

from app.models.schemas import Document, Evidence


def analyze_document(document: Document) -> Evidence:
    facts = {
        "date": document.date.isoformat(), "document_id": document.document_id,
        "document_type": document.document_type, "authority_type": document.authority_type,
        "supersedes": document.supersedes,
    }
    if document.service:
        facts["service"] = document.service
    if document.version:
        facts["version"] = document.version
    if "latency" in (document.content + " " + " ".join(document.claims)).lower():
        facts["symptom"] = "latency"
    text = " ".join(document.claims + [document.content])
    root_cause = re.search(r"caused by ([^.]+)", text, flags=re.IGNORECASE)
    if root_cause:
        facts["root_cause"] = root_cause.group(1).strip()
    events: list[dict[str, object]] = []
    if document.document_type == "deployment_note":
        events.append({
            "event": "deployment", "timestamp": document.timestamp.isoformat() if document.timestamp else None,
            "date": document.date.isoformat(), "source": document.document_id,
            "service": document.service, "version": document.version,
        })
    elif document.document_type == "incident_report":
        events.append({
            "event": "incident", "timestamp": document.timestamp.isoformat() if document.timestamp else None,
            "date": document.date.isoformat(), "source": document.document_id,
            "service": document.service, "version": document.version,
        })
    elif document.document_type == "postmortem":
        events.append({
            "event": "historical_incident", "timestamp": document.timestamp.isoformat() if document.timestamp else None,
            "date": document.date.isoformat(), "source": document.document_id,
            "service": document.service, "version": document.version,
        })
    return Evidence(document_id=document.document_id, document_type=document.document_type, facts=facts, claims=document.claims, events=events)
