from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.agent.models import AgentInvestigationRequest
from app.agent.service import run_reasoning_agent
from app.config import get_settings
from app.investigation.investigator import run_investigation
from app.ranking.evidence import rank_evidence
from app.search.bm25 import rebuild_bm25_index, search_bm25
from app.search.hybrid import hybrid_search
from app.search.indexing import reindex_all_documents
from app.search.semantic import semantic_search
from app.storage.postgres import get_db


router = APIRouter(tags=["semantic search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be empty")
        return value

    @field_validator("filters")
    @classmethod
    def filters_must_be_simple(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if value is None:
            return value
        allowed = {"services", "document_type", "versions", "incident_ids", "deployment_ids"}
        for key, filter_value in value.items():
            if key not in allowed:
                raise ValueError(f"Unsupported filter field: {key}")
            values = filter_value if isinstance(filter_value, list) else [filter_value]
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise ValueError(f"Filter '{key}' must contain non-empty string values")
        return value


class InvestigationSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    max_hops: int = Field(default_factory=lambda: get_settings().INVESTIGATION_MAX_HOPS, ge=1)
    results_per_hop: int = Field(default_factory=lambda: get_settings().INVESTIGATION_RESULTS_PER_HOP, ge=1)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query cannot be empty")
        return value

    @field_validator("max_hops")
    @classmethod
    def max_hops_must_be_bounded(cls, value: int) -> int:
        if value > get_settings().INVESTIGATION_MAX_HOPS:
            raise ValueError(f"max_hops must be between 1 and {get_settings().INVESTIGATION_MAX_HOPS}")
        return value

    @field_validator("results_per_hop")
    @classmethod
    def results_per_hop_must_be_bounded(cls, value: int) -> int:
        if value > get_settings().MAX_SEARCH_LIMIT:
            raise ValueError(f"results_per_hop must be between 1 and {get_settings().MAX_SEARCH_LIMIT}")
        return value


@router.post("/search", summary="Semantic search over indexed document chunks")
async def search(request: SearchRequest):
    try:
        results = semantic_search(request.query, limit=request.limit, filters=request.filters)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Semantic search unavailable: {exc}") from exc

    return {
        "query": request.query,
        "results": results,
    }


@router.post("/bm25-search", summary="BM25 lexical search over stored document chunks")
async def bm25_search(request: SearchRequest, db: Session = Depends(get_db)):
    try:
        results = search_bm25(request.query, limit=request.limit, filters=request.filters, db=db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"BM25 search unavailable: {exc}") from exc

    return {
        "query": request.query,
        "results": results,
    }


@router.post("/hybrid-search", summary="Hybrid BM25 and semantic search over document chunks")
async def hybrid_search_endpoint(request: SearchRequest, db: Session = Depends(get_db)):
    try:
        results = hybrid_search(
            request.query,
            limit=request.limit,
            filters=request.filters,
            bm25_searcher=lambda query, limit, filters=None: search_bm25(query, limit=limit, filters=filters, db=db),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Hybrid search unavailable: {exc}") from exc

    return {
        "query": request.query,
        "results": results,
    }


@router.post("/ranked-search", summary="Hybrid retrieval followed by deterministic evidence ranking")
async def ranked_search(request: SearchRequest, db: Session = Depends(get_db)):
    try:
        candidates = hybrid_search(
            request.query,
            limit=request.limit,
            filters=request.filters,
            bm25_searcher=lambda query, limit, filters=None: search_bm25(query, limit=limit, filters=filters, db=db),
        )
        results = rank_evidence(request.query, candidates, limit=request.limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ranked search unavailable: {exc}") from exc

    return {
        "query": request.query,
        "results": results,
    }


def ranked_search_results(query: str, limit: int, db: Session) -> list[dict]:
    candidates = hybrid_search(
        query,
        limit=limit,
        bm25_searcher=lambda search_query, limit, filters=None: search_bm25(search_query, limit=limit, filters=filters, db=db),
    )
    return rank_evidence(query, candidates, limit=limit)


@router.post("/investigation-search", summary="Multi-hop investigation retrieval over ranked evidence")
async def investigation_search(request: InvestigationSearchRequest, db: Session = Depends(get_db)):
    try:
        result = run_investigation(
            request.query,
            ranked_searcher=lambda query, limit: ranked_search_results(query, limit, db),
            max_hops=request.max_hops,
            results_per_hop=request.results_per_hop,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Investigation search unavailable: {exc}") from exc

    return result.model_dump()


@router.post("/investigate", summary="Evidence-backed investigation reasoning over multi-hop retrieval")
async def investigate(request: AgentInvestigationRequest, db: Session = Depends(get_db)):
    try:
        result = run_reasoning_agent(
            query=request.query,
            ranked_searcher=lambda query, limit: ranked_search_results(query, limit, db),
            max_hops=request.max_hops,
            results_per_hop=request.results_per_hop,
            include_trace=request.include_trace,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Investigation reasoning unavailable: {exc}") from exc

    if result.status == "llm_unavailable":
        raise HTTPException(status_code=503, detail=result.model_dump())
    if result.status == "validation_failed":
        raise HTTPException(status_code=422, detail=result.model_dump())
    return result.model_dump()


@router.post("/admin/reindex", summary="Rebuild Qdrant vectors from PostgreSQL chunks")
async def admin_reindex(db: Session = Depends(get_db)):
    return {
        "success": True,
        "reindex": reindex_all_documents(db),
    }


@router.post("/admin/reindex-bm25", summary="Rebuild BM25 index from PostgreSQL chunks")
async def admin_reindex_bm25(db: Session = Depends(get_db)):
    return {
        "success": True,
        "reindex": rebuild_bm25_index(db),
    }
