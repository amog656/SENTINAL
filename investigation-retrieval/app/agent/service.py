from typing import Callable, Optional

from app.agent.llm import LLMProvider, LLMUnavailableError, create_llm_provider
from app.agent.models import AgentResult
from app.agent.prompts import SYSTEM_PROMPT, reasoning_user_prompt
from app.agent.reasoning import (
    ReasoningValidationError,
    build_evidence_packet,
    unavailable_analysis,
    validate_reasoning_output,
    validation_failed_analysis,
)
from app.config import get_settings
from app.investigation.investigator import RankedSearch, run_investigation
from app.investigation.models import InvestigationResult


InvestigationRunner = Callable[[str, RankedSearch, Optional[int], Optional[int]], InvestigationResult]


def run_reasoning_agent(
    query: str,
    ranked_searcher: RankedSearch,
    max_hops: Optional[int] = None,
    results_per_hop: Optional[int] = None,
    include_trace: bool = True,
    llm_provider: Optional[LLMProvider] = None,
    investigation_runner: Callable[..., InvestigationResult] = run_investigation,
) -> AgentResult:
    investigation = investigation_runner(
        query,
        ranked_searcher=ranked_searcher,
        max_hops=max_hops,
        results_per_hop=results_per_hop,
    )
    if not include_trace:
        investigation.trace = []

    evidence_packet = build_evidence_packet(investigation, max_evidence=get_settings().AGENT_MAX_EVIDENCE)
    provider = llm_provider or create_llm_provider()
    try:
        raw = provider.generate(SYSTEM_PROMPT, reasoning_user_prompt(query, evidence_packet.text))
    except LLMUnavailableError:
        analysis = unavailable_analysis()
        return AgentResult(
            status="llm_unavailable",
            query=query,
            analysis=analysis,
            investigation=investigation,
            evidence_packet=evidence_packet,
            message="Investigation retrieval completed, but no reasoning provider is configured.",
        )

    try:
        analysis = validate_reasoning_output(raw, evidence_packet)
    except ReasoningValidationError as exc:
        analysis = validation_failed_analysis(exc.errors)
        return AgentResult(
            status="validation_failed",
            query=query,
            analysis=analysis,
            investigation=investigation,
            evidence_packet=evidence_packet,
            message="LLM response failed validation.",
        )

    return AgentResult(
        status=analysis.status,
        query=query,
        analysis=analysis,
        investigation=investigation,
        evidence_packet=evidence_packet,
    )
