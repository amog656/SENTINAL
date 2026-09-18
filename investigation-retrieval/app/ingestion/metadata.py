import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from app.models.document import ExtractedMetadata


DEFAULT_KNOWN_SERVICES = [
    "orders-api",
    "catalog-api",
    "payment-service",
    "inventory-service",
]

TECHNICAL_VOCABULARY = [
    "connection pool exhaustion",
    "schema migration",
    "disk saturation",
    "load balancer",
    "api gateway",
    "memory leak",
    "cpu spike",
    "http 500",
    "http 502",
    "http 503",
    "connection pool",
    "postgresql",
    "database",
    "mysql",
    "redis",
    "kafka",
    "migration",
    "latency",
    "timeout",
    "cpu",
    "memory",
    "disk",
    "cache",
    "deadlock",
    "replication",
    "queue",
    "kubernetes",
    "docker",
]

DOCUMENT_TYPES = {
    "incident_report": ["incident report", "incident", "inc-"],
    "deployment_note": ["deployment note", "deployment", "deploy", "dep-"],
    "postmortem": ["postmortem", "post-mortem", "root cause"],
    "architecture_document": ["architecture", "design document", "system design"],
    "troubleshooting_guide": ["troubleshooting", "runbook", "playbook"],
    "customer_complaint": ["customer complaint", "complaint"],
    "engineering_discussion": ["engineering discussion", "slack", "discussion"],
}

MONTH_DATE_FORMATS = ["%B %d, %Y", "%b %d, %Y"]


def _unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    unique = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def get_known_services() -> List[str]:
    configured = os.getenv("KNOWN_SERVICES")
    if not configured:
        return DEFAULT_KNOWN_SERVICES
    return [service.strip() for service in configured.split(",") if service.strip()]


def extract_services(text: str, known_services: Optional[List[str]] = None) -> List[str]:
    services = known_services or get_known_services()
    found = []
    for service in services:
        pattern = re.compile(rf"(?<![\w-]){re.escape(service)}(?![\w-])", re.IGNORECASE)
        if pattern.search(text):
            found.append(service)
    return _unique_preserve_order(found)


def extract_versions(text: str) -> List[str]:
    pattern = re.compile(r"(?<![\w.])v?\d+(?:\.\d+){0,2}(?![\w.])", re.IGNORECASE)
    values = []
    for match in pattern.finditer(text):
        value = match.group(0)
        if not value.lower().startswith("v") and value.count(".") < 2:
            continue
        values.append(value)
    return _unique_preserve_order(values)


def extract_incident_ids(text: str) -> List[str]:
    return _unique_preserve_order(match.upper() for match in re.findall(r"\bINC-\d+\b", text, flags=re.IGNORECASE))


def extract_deployment_ids(text: str) -> List[str]:
    return _unique_preserve_order(match.upper() for match in re.findall(r"\bDEP-\d+\b", text, flags=re.IGNORECASE))


def extract_dates(text: str) -> List[str]:
    dates = []

    for match in re.findall(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b", text):
        normalized = match.replace("/", "-")
        try:
            dates.append(datetime.strptime(normalized, "%Y-%m-%d").date().isoformat())
        except ValueError:
            continue

    for match in re.findall(r"\b\d{2}-\d{2}-\d{4}\b", text):
        try:
            dates.append(datetime.strptime(match, "%d-%m-%Y").date().isoformat())
        except ValueError:
            continue

    month_pattern = r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b"
    for match in re.findall(month_pattern, text, flags=re.IGNORECASE):
        normalized = re.sub(r"^Sept\b", "Sep", match, flags=re.IGNORECASE)
        for fmt in MONTH_DATE_FORMATS:
            try:
                dates.append(datetime.strptime(normalized.title(), fmt).date().isoformat())
                break
            except ValueError:
                continue

    return _unique_preserve_order(dates)


def detect_document_type(filename: str, text: str, source_metadata: Optional[Dict[str, Any]] = None) -> str:
    source_metadata = source_metadata or {}
    for key in ("document_type", "type"):
        value = source_metadata.get(key)
        if isinstance(value, str) and value in DOCUMENT_TYPES:
            return value

    haystacks = [
        filename.lower(),
        filename.replace("_", " ").lower(),
        text[:2000].lower(),
    ]
    for document_type, hints in DOCUMENT_TYPES.items():
        for hint in hints:
            if any(hint in haystack for haystack in haystacks):
                return document_type
    return "unknown"


def extract_technical_entities(text: str, vocabulary: Optional[List[str]] = None) -> List[str]:
    vocabulary = vocabulary or TECHNICAL_VOCABULARY
    found = []
    for entity in vocabulary:
        pattern = re.compile(rf"(?<![\w-]){re.escape(entity)}(?![\w-])", re.IGNORECASE)
        if pattern.search(text):
            found.append(entity)
    return _unique_preserve_order(found)


def extract_metadata(filename: str, text: str, source_metadata: Optional[Dict[str, Any]] = None) -> ExtractedMetadata:
    return ExtractedMetadata(
        services=extract_services(text),
        versions=extract_versions(text),
        dates=extract_dates(text),
        document_type=detect_document_type(filename, text, source_metadata),
        incident_ids=extract_incident_ids(text),
        deployment_ids=extract_deployment_ids(text),
        technical_entities=extract_technical_entities(text),
    )
