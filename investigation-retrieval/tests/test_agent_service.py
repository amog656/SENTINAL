import json

from app.agent.llm import LLMUnavailableError
from app.agent.service import run_reasoning_agent
from app.investigation.models import EvidenceItem, InvestigationResult


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return json.dumps(self.payload)


class MissingLLM:
    def generate(self, system_prompt, user_prompt):
        raise LLMUnavailableError("missing")


def investigation_runner(query, ranked_searcher, max_hops=None, results_per_hop=None):
    return InvestigationResult(
        query=query,
        max_hops=max_hops or 1,
        hops_completed=0,
        evidence=[
            EvidenceItem(
                chunk_id="C1",
                document_id="D1",
                text="orders-api latency increased.",
                final_score=0.9,
                retrieval_query=query,
            )
        ],
        clues=[],
        relationships=[],
        hops=[],
        trace=[],
        summary={"documents_examined": 1, "chunks_examined": 1, "hops_completed": 0},
    )


def test_reasoning_agent_invokes_retrieval_and_mock_llm():
    llm = FakeLLM(
        {
            "status": "completed",
            "summary": "Latency was observed.",
            "confidence": "medium",
            "findings": [{"claim": "orders-api latency increased.", "type": "observed_fact", "evidence_ids": ["E1"]}],
            "timeline": [],
            "causal_chain": [],
            "contradictions": [],
            "hypotheses": [],
            "missing_evidence": [],
            "conclusion": "Evidence supports latency.",
            "evidence_used": ["E1"],
        }
    )

    result = run_reasoning_agent(
        "Why did orders-api slow?",
        ranked_searcher=lambda query, limit: [],
        max_hops=1,
        results_per_hop=1,
        llm_provider=llm,
        investigation_runner=investigation_runner,
    )

    assert result.status == "completed"
    assert result.analysis.evidence_coverage == 1.0
    assert "EVIDENCE [E1]" in llm.calls[0][1]


def test_reasoning_agent_returns_structured_llm_unavailable_result():
    result = run_reasoning_agent(
        "Why did orders-api slow?",
        ranked_searcher=lambda query, limit: [],
        max_hops=1,
        results_per_hop=1,
        llm_provider=MissingLLM(),
        investigation_runner=investigation_runner,
    )

    assert result.status == "llm_unavailable"
    assert result.investigation.evidence[0].chunk_id == "C1"


def test_reasoning_agent_returns_validation_failed_result():
    result = run_reasoning_agent(
        "Why did orders-api slow?",
        ranked_searcher=lambda query, limit: [],
        max_hops=1,
        results_per_hop=1,
        llm_provider=FakeLLM({"not": "valid"}),
        investigation_runner=investigation_runner,
    )

    assert result.status == "validation_failed"
    assert result.analysis.validation_errors
