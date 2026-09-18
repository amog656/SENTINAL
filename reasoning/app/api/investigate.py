from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import ContinueInvestigationRequest, InvestigateRequest, InvestigationStarted

router = APIRouter(tags=["investigation"])


@router.post("/investigate", response_model=InvestigationStarted, status_code=202)
async def investigate(request_body: InvestigateRequest, request: Request) -> InvestigationStarted:
    registry = request.app.state.registry
    state = request.app.state.investigator.new_state(request_body.question)
    await registry.create(state)
    asyncio.create_task(registry.run(state.investigation_id, request.app.state.investigator))
    return InvestigationStarted(investigation_id=state.investigation_id)


@router.post("/continue-investigation", response_model=InvestigationStarted, status_code=202)
async def continue_investigation(body: ContinueInvestigationRequest, request: Request) -> InvestigationStarted:
    # TODO(Phase 2): continue an existing notebook with new hops rather than restart.
    prior = await request.app.state.registry.get(body.investigation_id)
    if not prior:
        raise HTTPException(404, "Unknown investigation_id")
    return await investigate(InvestigateRequest(question=f"{prior.original_question}\nFollow-up: {body.instruction}"), request)
