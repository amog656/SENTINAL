from __future__ import annotations

import re
import uuid
from collections.abc import Awaitable, Callable

from app.agent.planner import create_plan, decode_follow_up, encode_follow_up, next_query_fallback
from app.llm.client import CachedLLMClient
from app.models.schemas import Document, Evidence, InvestigationEvent, InvestigationState, SearchRecord
from app.reasoning.clue_extractor import extract_clues
from app.reasoning.confidence import calculate_confidence
from app.reasoning.contradiction import detect_contradictions, resolve_contradiction
from app.reasoning.evidence_analyzer import analyze_document
from app.reasoning.similarity import compare_incidents, unperformable_comparison
from app.reasoning.sufficiency import assess_sufficiency
from app.retrieval.interface import RetrievalClient
from app.report.generator import generate_report
from app.timeline.builder import build_timeline


EventSink = Callable[[InvestigationEvent], Awaitable[None]]


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


class Investigator:
    def __init__(self, retrieval: RetrievalClient, llm: CachedLLMClient):
        self.retrieval, self.llm = retrieval, llm

    def new_state(self, question: str) -> InvestigationState:
        return InvestigationState(investigation_id=str(uuid.uuid4()), original_question=question)

    async def _emit(self, state: InvestigationState, sink: EventSink, step_type: str, description: str, payload: dict) -> None:
        event = InvestigationEvent(investigation_id=state.investigation_id, step_type=step_type, description=description, payload=payload)
        state.investigation_steps.append(event)
        await sink(event)

    async def _set_goal(self, state: InvestigationState, sink: EventSink, goal_id: str, status: str, evidence_ids: list[str]) -> None:
        goal = next(goal for goal in state.goals if goal.id == goal_id)
        previous = goal.status
        goal.status, goal.evidence_ids = status, list(dict.fromkeys(evidence_ids))
        if previous != status:
            await self._emit(state, sink, "goal_status_changed", f"Goal '{goal.description}' is now {status}", {
                "goal_id": goal.id, "previous_status": previous, "status": status,
                "evidence_ids": goal.evidence_ids,
            })

    async def _search(self, state: InvestigationState, sink: EventSink, query: str) -> list[Document]:
        normalized = _normalize_query(query)
        if any(_normalize_query(record.query) == normalized for record in state.search_history):
            return []
        await self._emit(state, sink, "search_issued", f"Searching RETRIEVAL: {query}", {"query": query})
        documents = await self.retrieval.search(query, {"limit": 1})
        state.search_history.append(SearchRecord(query=query, filters={"limit": 1}, result_document_ids=[d.document_id for d in documents]))
        return documents

    async def _analyze(self, state: InvestigationState, sink: EventSink, documents: list[Document]) -> list[Evidence]:
        added: list[Evidence] = []
        for document in documents:
            if document.document_id in state.documents_seen:
                continue
            state.documents_seen.append(document.document_id)
            evidence = analyze_document(document)
            state.evidence.append(evidence)
            added.append(evidence)
            if document.service and document.service not in state.services:
                state.services.append(document.service)
            if document.version and document.version not in state.versions:
                state.versions.append(document.version)
            if document.document_type == "incident_report":
                state.incidents.append(document.document_id)
            if document.document_type == "deployment_note":
                state.deployments.append(document.document_id)
            await self._emit(state, sink, "document_analyzed", f"Analyzed {document.document_id}", {"document_id": document.document_id, "facts": evidence.facts})
            try:
                await self.llm.complete(f"extract clues\n{document.document_id}\n{document.content}", lambda: ",".join(extract_clues(evidence)))
            except Exception as exc:
                state.open_questions.append(f"Semantic clue pass skipped: {exc}")
            new_clues = [clue for clue in extract_clues(evidence) if clue not in state.extracted_clues]
            state.extracted_clues.extend(new_clues)
            await self._emit(state, sink, "clue_extracted", f"Extracted {len(new_clues)} usable clues from {document.document_id}", {"document_id": document.document_id, "clues": new_clues})
        return added

    async def _update_goals_from_evidence(self, state: InvestigationState, sink: EventSink) -> None:
        current = next((item for item in state.evidence if item.document_type == "incident_report"), None)
        deployment = next((item for item in state.evidence if item.document_type == "deployment_note"), None)
        historical = next((item for item in state.evidence if item.document_type == "postmortem"), None)
        if current:
            await self._set_goal(state, sink, "current_incident", "supported", [current.document_id])
        if current and deployment:
            same_entity = current.facts.get("service") == deployment.facts.get("service") and current.facts.get("version") == deployment.facts.get("version")
            deploy_time = deployment.events[0].get("timestamp") or deployment.facts.get("date")
            incident_time = current.events[0].get("timestamp") or current.facts.get("date")
            if same_entity and str(deploy_time) < str(incident_time):
                relationship = {"relationship_type": "temporal_precedence", "source_document_ids": [deployment.document_id, current.document_id], "causation": "not_established"}
                if relationship not in state.hypotheses:
                    state.hypotheses.append(relationship)
                await self._set_goal(state, sink, "deployment_relationship", "supported", [deployment.document_id, current.document_id])
        if current and historical and current.facts.get("service") == historical.facts.get("service") and current.facts.get("symptom") == historical.facts.get("symptom"):
            await self._set_goal(state, sink, "historical_incident", "supported", [current.document_id, historical.document_id])
        if any(goal.id == "guidance_evidence" for goal in state.goals):
            guidance_ids = [item.document_id for item in state.evidence if item.document_type == "troubleshooting_guide"]
            if guidance_ids:
                await self._set_goal(state, sink, "guidance_evidence", "supported", guidance_ids)

    async def _process_contradictions(self, state: InvestigationState, sink: EventSink) -> None:
        """Use only already-retrieved guidance; follow-up selection remains in the shared hop loop."""
        findings = detect_contradictions(state.evidence)
        if not findings:
            return
        evidence_by_id = {item.document_id: item for item in state.evidence}
        for finding in findings:
            if any(existing["documents"] == finding["documents"] for existing in state.contradictions):
                continue
            state.contradictions.append(finding)
            await self._emit(state, sink, "contradiction_found", "Detected conflicting troubleshooting guidance", finding)
            resolution = resolve_contradiction(finding, evidence_by_id)
            state.resolved_conflicts.append(resolution)
            await self._emit(state, sink, "conflict_resolved", resolution["reason"], resolution)
            if any(goal.id == "guidance_evidence" for goal in state.goals):
                await self._set_goal(state, sink, "guidance_evidence", "supported", finding["source_document_ids"])
                # The goal is to investigate the conflict. A documented,
                # attempted-but-inconclusive resolution is still a completed
                # investigation result—not an automatic global evidence failure.
                await self._set_goal(state, sink, "guidance_conflict", "supported" if resolution["resolution_attempted"] else "unresolved", finding["source_document_ids"])
            if resolution["resolution_status"] == "unresolved":
                question = "Which troubleshooting context applies to the current situation?"
                if question not in state.open_questions:
                    state.open_questions.append(question)

    async def _refresh_timeline_and_confidence(self, state: InvestigationState, sink: EventSink) -> None:
        timeline = build_timeline(state.evidence)
        if timeline != state.timeline:
            state.timeline = timeline
            state.technical_events = timeline
            await self._emit(state, sink, "timeline_updated", "Updated source-traceable chronological timeline", {"timeline": timeline})
        state.confidence, state.confidence_factors = calculate_confidence(state)
        await self._emit(state, sink, "confidence_updated", "Updated deterministic evidence confidence", {"confidence": state.confidence, "factors": state.confidence_factors})

    async def _compare_history_if_ready(self, state: InvestigationState, sink: EventSink) -> None:
        if state.historical_comparisons:
            return
        current = next((item for item in state.evidence if item.document_type == "incident_report"), None)
        historical = next((item for item in state.evidence if item.document_type == "postmortem"), None)
        if not current:
            return
        historical_goal = next(goal for goal in state.goals if goal.id == "historical_incident")
        if historical:
            comparison = compare_incidents(current, historical)
        elif historical_goal.status == "unsupported":
            comparison = unperformable_comparison(current, "No comparable historical incident was found after the targeted historical search.")
        else:
            return
        state.historical_comparisons.append(comparison)
        await self._emit(state, sink, "historical_comparison", comparison["conclusion"], comparison)
        # This goal asks whether comparison was performed—not whether it found
        # identical root causes. A conservative negative conclusion can satisfy it.
        status = "supported" if comparison["comparison_performed"] else "unsupported"
        await self._set_goal(state, sink, "historical_comparison", status, comparison["source_document_ids"])
        if comparison["comparison_performed"] and comparison["same_root_cause"] == "not_established":
            question = f"Whether the current incident has the same root cause as {historical.document_id}."
            if question not in state.open_questions:
                state.open_questions.append(question)

    async def _next_follow_up(self, state: InvestigationState) -> tuple[str, str] | None:
        # Every candidate, including an unresolved comparison, is asked to name
        # its target goal. A missing query is the explicit no_targetable_goal
        # signal; the loop never stops merely because the model "feels done".
        candidates = [goal for goal in state.goals if goal.status in {"not_investigated", "unresolved"}]
        if not candidates:
            return None
        goal = candidates[0]
        service = state.services[0] if state.services else None
        version = state.versions[0] if state.versions else None
        symptom = next((clue for clue in state.extracted_clues if clue in {"latency", "timeout", "error"}), None)
        fallback_query = next_query_fallback(goal.id, service, version, symptom)
        try:
            response = await self.llm.complete(
                f"phase2 follow-up\ngoal={goal.id}\nservice={service}\nversion={version}\nsymptom={symptom}",
                lambda: encode_follow_up(goal.id, fallback_query),
            )
            proposed_goal, query = decode_follow_up(response)
        except Exception as exc:
            state.open_questions.append(f"Follow-up proposal skipped: {exc}")
            return None
        if proposed_goal != goal.id or not query or not query.strip():
            return None
        if any(_normalize_query(record.query) == _normalize_query(query) for record in state.search_history):
            return None
        return goal.id, query.strip()

    async def run(self, state: InvestigationState, sink: EventSink) -> dict:
        plan = create_plan(state.original_question)
        state.investigation_plan = plan
        state.goals = [goal.model_copy(deep=True) for goal in plan.goals]
        try:
            await self.llm.complete(f"plan investigation\n{state.original_question}", lambda: plan.initial_query)
        except Exception as exc:
            state.open_questions.append(f"Planner semantic pass skipped: {exc}")
        await self._emit(state, sink, "plan_created", "Created a bounded goal-aware Phase-2 investigation plan", plan.model_dump())

        documents = await self._search(state, sink, plan.initial_query)
        await self._analyze(state, sink, documents)
        await self._update_goals_from_evidence(state, sink)
        await self._process_contradictions(state, sink)
        await self._refresh_timeline_and_confidence(state, sink)

        while len(state.search_history) < plan.max_hops:
            proposal = await self._next_follow_up(state)
            if proposal is None:
                state.loop_termination_reason = "goals_covered" if all(goal.status == "supported" for goal in state.goals if goal.critical_to_question) else "no_targetable_goal"
                break
            goal_id, query = proposal
            await self._emit(state, sink, "follow_up_search", f"Following goal '{goal_id}' with: {query}", {"goal_id": goal_id, "query": query, "clues": state.extracted_clues})
            documents = await self._search(state, sink, query)
            added = await self._analyze(state, sink, documents)
            if not added:
                await self._set_goal(state, sink, goal_id, "unsupported", [])
            await self._update_goals_from_evidence(state, sink)
            await self._process_contradictions(state, sink)
            await self._compare_history_if_ready(state, sink)
            await self._refresh_timeline_and_confidence(state, sink)
        else:
            state.loop_termination_reason = "max_hops_reached"

        await self._compare_history_if_ready(state, sink)
        await self._set_goal(state, sink, "evidence_gaps", "supported", [item.document_id for item in state.evidence])
        await self._refresh_timeline_and_confidence(state, sink)
        state.status, sufficiency_questions = assess_sufficiency(state)
        state.evidence_status = state.status
        state.open_questions.extend(question for question in sufficiency_questions if question not in state.open_questions)
        await self._emit(state, sink, "sufficiency_checked", f"Evidence status: {state.status}", {"status": state.status, "open_questions": state.open_questions, "loop_termination_reason": state.loop_termination_reason})
        await self._emit(state, sink, "report_ready", "Investigation report is ready", {"status": state.status, "confidence": state.confidence})
        return generate_report(state)
