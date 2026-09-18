from __future__ import annotations

from app.models.schemas import Evidence


def _restart_polarity(claim: str) -> str | None:
    text = claim.lower()
    if "restart" not in text:
        return None
    if "do not restart" in text or "don't restart" in text:
        return "prohibit"
    if "restart" in text:
        return "recommend"
    return None


def _context(claim: str) -> str | None:
    text = claim.lower()
    if "dependency" in text:
        return "dependency_failure"
    if "latency" in text:
        return "persistent_latency"
    return None


def classify_guidance_pair(left: Evidence, right: Evidence, left_claim: str, right_claim: str) -> dict:
    """Classify policy relationship, including compatible/apparent cases."""
    left_polarity, right_polarity = _restart_polarity(left_claim), _restart_polarity(right_claim)
    left_context, right_context = _context(left_claim), _context(right_claim)
    base = {
        "claims": [{"document_id": left.document_id, "claim": left_claim}, {"document_id": right.document_id, "claim": right_claim}],
        "documents": [left.document_id, right.document_id], "source_document_ids": [left.document_id, right.document_id],
        "contexts": [left_context, right_context],
    }
    if not left_polarity or not right_polarity:
        return {**base, "contradiction": False, "type": "no_contradiction"}
    if left_polarity == right_polarity:
        return {**base, "contradiction": False, "type": "apparent" if left_context != right_context else "no_contradiction"}
    return {**base, "contradiction": True, "type": "contextual" if left_context != right_context and (left_context or right_context) else "direct", "recommendations": [left_polarity, right_polarity]}


def detect_contradictions(evidence: list[Evidence]) -> list[dict]:
    """Detect policy tension semantically enough for the MVP, not by ID/string pairs."""
    guides = [item for item in evidence if item.document_type == "troubleshooting_guide"]
    findings: list[dict] = []
    for index, left in enumerate(guides):
        for right in guides[index + 1:]:
            for left_claim in left.claims:
                for right_claim in right.claims:
                    classification = classify_guidance_pair(left, right, left_claim, right_claim)
                    if classification["contradiction"]:
                        findings.append(classification)
    return findings


def resolve_contradiction(finding: dict, evidence_by_id: dict[str, Evidence], *, attempt: bool = True) -> dict:
    """Apply deterministic applicability/supersession rules; never use date alone."""
    base = {**finding, "resolution_attempted": attempt}
    if not attempt:
        return {**base, "resolution_status": "unresolved", "reason": "Resolution was not attempted before the investigation budget ended."}

    left, right = (evidence_by_id[doc_id] for doc_id in finding["documents"])
    left_context, right_context = finding["contexts"]
    # A generic recommendation and a dependency-specific prohibition can both be
    # valid. The newer guide is useful only because the context is applicable;
    # explicit supersession is corroboration, never the sole deciding rule.
    if finding["type"] == "contextual" and left_context != right_context:
        dependency = left if left_context == "dependency_failure" else right
        generic = right if dependency is left else left
        return {
            **base,
            "resolution_status": "resolved",
            "selected_document_id": dependency.document_id,
            "rejected_document_id": None,
            "reason": f"Both recommendations apply in different contexts: {dependency.document_id} governs dependency failures; {generic.document_id} remains applicable outside that context."
            + (" Explicit supersession supports this scoped interpretation." if dependency.facts.get("supersedes") == generic.document_id else ""),
            "rule_used": "context_specific_guidance_with_explicit_supersession",
            "applicability": {dependency.document_id: "dependency_failure", generic.document_id: "persistent_latency_without_dependency_failure"},
        }
    if left.facts.get("supersedes") == right.document_id or right.facts.get("supersedes") == left.document_id:
        selected, rejected = (left, right) if left.facts.get("supersedes") == right.document_id else (right, left)
        return {
            **base,
            "resolution_status": "resolved", "selected_document_id": selected.document_id,
            "rejected_document_id": rejected.document_id,
            "reason": "The selected guidance explicitly supersedes the competing guidance for the same applicable context.",
            "rule_used": "explicit_supersession", "applicability": {},
        }
    return {
        **base,
        "resolution_status": "unresolved", "selected_document_id": None,
        "rejected_document_id": None,
        "reason": "The recommendations conflict, but the available evidence does not establish which context applies.",
        "rule_used": "insufficient_applicability_context", "applicability": {},
    }
