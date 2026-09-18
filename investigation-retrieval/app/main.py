import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.api.search import router as search_router
from app.api.upload import router as upload_router
from app.storage.postgres import SessionLocal, init_db
from app.storage.qdrant import check_qdrant_connection


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
    except Exception as exc:
        logger.warning("Database initialization skipped: %s", exc)
    yield


app = FastAPI(
    title="Investigation Retrieval Engine",
    description="Evidence retrieval backend for the Investigation Agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(upload_router)
app.include_router(search_router)


@app.get("/")
async def root():
    return {
        "service": "Investigation Retrieval Engine",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    postgres_status = "unhealthy"
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        postgres_status = "healthy"
    except Exception as exc:
        logger.warning("PostgreSQL health check failed: %s", exc)

    qdrant_status = "healthy" if check_qdrant_connection() else "unhealthy"

    return {
        "status": "healthy",
        "postgres": postgres_status,
        "qdrant": qdrant_status,
    }
