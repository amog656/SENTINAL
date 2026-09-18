from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import TimelineRequest, TimelineResponse

router = APIRouter(tags=["on-demand analysis"])


@router.post("/timeline", response_model=TimelineResponse)
async def timeline(body: TimelineRequest) -> dict:
    events = [{"event": document.document_type, "timestamp": document.timestamp.isoformat() if document.timestamp else None, "date": document.date.isoformat(), "source": document.document_id} for document in body.documents]
    return {"timeline": sorted(events, key=lambda event: (event["timestamp"] or event["date"], event["source"]))}
