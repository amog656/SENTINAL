from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.models.schemas import InvestigationReport, InvestigationStepsResponse

router = APIRouter(tags=["investigation"])


@router.get("/investigate/{investigation_id}/stream")
async def stream_investigation(investigation_id: str, request: Request) -> StreamingResponse:
    registry = request.app.state.registry
    if not await registry.get(investigation_id):
        raise HTTPException(404, "Unknown investigation_id")

    async def events():
        async for event in registry.subscribe(investigation_id):
            yield f"event: investigation_step\ndata: {event.model_dump_json()}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/investigate/{investigation_id}/steps", response_model=InvestigationStepsResponse)
async def polling_steps(investigation_id: str, request: Request) -> dict:
    state = await request.app.state.registry.get(investigation_id)
    if not state:
        raise HTTPException(404, "Unknown investigation_id")
    return {"investigation_id": investigation_id, "status": state.status, "steps": [item.model_dump() for item in state.investigation_steps]}


@router.get("/investigate/{investigation_id}/report", response_model=InvestigationReport)
async def report(investigation_id: str, request: Request) -> dict:
    result = await request.app.state.registry.report(investigation_id)
    if result is None:
        raise HTTPException(404, "Unknown investigation_id")
    return result
