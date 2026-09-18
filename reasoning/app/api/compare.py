from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import CompareIncidentsRequest, IncidentComparisonResponse

router = APIRouter(tags=["on-demand analysis"])


@router.post("/compare-incidents", response_model=IncidentComparisonResponse)
async def compare_incidents(body: CompareIncidentsRequest) -> dict:
    same_service = body.current.service == body.historical.service and body.current.service is not None
    current_text = " ".join(body.current.claims + [body.current.content]).lower()
    historical_text = " ".join(body.historical.claims + [body.historical.content]).lower()
    same_symptom = "latency" in current_text and "latency" in historical_text
    return {"current_document_id": body.current.document_id, "historical_document_id": body.historical.document_id, "similarity": "medium" if same_service and same_symptom else "low", "conclusion": "Similar symptom, but same root cause is not established.", "source_document_ids": [body.current.document_id, body.historical.document_id]}
