from __future__ import annotations

from pathlib import Path


MAX_HOPS = 5
MAX_RESULTS_PER_SEARCH = 1
AUTHORITY_ORDER = {
    "postmortem": 4,
    "deployment_review": 3,
    "troubleshooting": 2,
    "discussion": 1,
}

# Named, deterministic Phase-1 confidence formula. Future phases can tune weights here.
CONFIDENCE_BASE = 0.20
CONFIDENCE_PER_SUPPORTING_DOCUMENT = 0.12
CONFIDENCE_PER_DOCUMENT_TYPE = 0.06
CONFIDENCE_CROSS_DOCUMENT_CONFIRMATION = 0.10
CONFIDENCE_TEMPORAL_CONSISTENCY_BONUS = 0.04
CONFIDENCE_ENTITY_CONSISTENCY_BONUS = 0.04
CONFIDENCE_UNRESOLVED_CONTRADICTION_PENALTY = 0.15
CONFIDENCE_MISSING_CRITICAL_EVIDENCE_PENALTY = 0.10
CONFIDENCE_UNSUPPORTED_ASSUMPTION_PENALTY = 0.05
INSUFFICIENT_CONFIDENCE_THRESHOLD = 0.40
MIN_SUPPORTING_DOCUMENTS = 2

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PACKAGE_ROOT / "data" / "documents.json"
CACHE_DIR = PACKAGE_ROOT / ".llm_cache"
LLM_CACHE_MODEL = "demo-deterministic-phase2"  # Versions Phase-2 cache keys; Phase-1 entries are ignored.
