# Investigation Retrieval Engine

## Overview

The **Investigation Retrieval Engine** is the Retrieval Intelligence layer for an AI-powered software incident investigation system. It provides the backend foundation for finding and connecting evidence across diverse sources including incident reports, deployment notes, postmortems, architecture documents, troubleshooting guides, customer complaints, engineering discussions, and Git/deployment information.

## Current Architecture

```
investigation-retrieval/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application with / and /health endpoints
│   ├── config.py         # Centralized configuration management
│   └── storage/
│       ├── __init__.py
│       ├── postgres.py   # SQLAlchemy database session management
│       └── qdrant.py     # Qdrant vector database client
├── data/
│   └── documents/        # Document storage (future use)
├── tests/
│   └── test_main.py      # Basic endpoint tests
├── docker-compose.yml    # PostgreSQL 16 + Qdrant infrastructure
├── Dockerfile            # FastAPI application container
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
└── .gitignore
```

## Requirements

- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 16 (via Docker)
- Qdrant (via Docker)

## Quick Start

### 1. Start Infrastructure

```bash
docker compose up -d
```

This starts:
- **PostgreSQL 16** on port 5432 (database: `investigation`, user: `investigator`)
- **Qdrant** on ports 6333 (HTTP) and 6334 (gRPC)

### 2. Configure Environment

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` if you need to customize connection settings.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run FastAPI Application

```bash
uvicorn app.main:app --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger Documentation**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service status |
| GET | `/health` | Health check |

### Example Responses

**GET /**
```json
{
  "service": "Investigation Retrieval Engine",
  "status": "running"
}
```

**GET /health**
```json
{
  "status": "healthy"
}
```

## Running Tests

```bash
pytest tests/
```

## What's Next (Phase 2+)

- Document ingestion pipeline (PDF, DOCX, TXT, JSON parsing)
- Text cleaning and chunking
- Metadata extraction
- Embedding generation (sentence-transformers)
- Qdrant vector storage and collections
- BM25 keyword search (rank-bm25)
- Metadata filtering
- Hybrid retrieval (vector + keyword)
- Evidence ranking
- Multi-hop retrieval
- Retrieval APIs
- Retrieval evaluation framework

## Configuration

All configuration is centralized in `app/config.py` and uses environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| POSTGRES_HOST | localhost | PostgreSQL host |
| POSTGRES_PORT | 5432 | PostgreSQL port |
| POSTGRES_USER | investigator | PostgreSQL username |
| POSTGRES_PASSWORD | investigator | PostgreSQL password |
| POSTGRES_DB | investigation | PostgreSQL database name |
| QDRANT_HOST | localhost | Qdrant host |
| QDRANT_PORT | 6333 | Qdrant HTTP port |