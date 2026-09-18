from __future__ import annotations

import asyncio
from collections import defaultdict

from fastapi import FastAPI

from app.api import compare, investigate, stream, timeline
from app.config import CACHE_DIR, DATA_PATH
from app.llm.cache import DiskLLMCache
from app.llm.client import CachedLLMClient
from app.models.schemas import InvestigationEvent, InvestigationState
from app.agent.investigator import Investigator
from app.retrieval.mock_client import MockRetrievalClient


class InvestigationRegistry:
    def __init__(self):
        self.states: dict[str, InvestigationState] = {}
        self.reports: dict[str, dict] = {}
        self.queues: dict[str, list[asyncio.Queue[InvestigationEvent | None]]] = defaultdict(list)

    async def create(self, state: InvestigationState) -> None:
        self.states[state.investigation_id] = state

    async def get(self, investigation_id: str) -> InvestigationState | None:
        return self.states.get(investigation_id)

    async def emit(self, event: InvestigationEvent) -> None:
        for queue in list(self.queues[event.investigation_id]):
            await queue.put(event)

    async def run(self, investigation_id: str, investigator: Investigator) -> None:
        state = self.states[investigation_id]
        try:
            self.reports[investigation_id] = await investigator.run(state, self.emit)
        finally:
            for queue in list(self.queues[investigation_id]):
                await queue.put(None)

    async def subscribe(self, investigation_id: str):
        state = self.states[investigation_id]
        for existing in state.investigation_steps:
            yield existing
        if state.status != "in_progress":
            return
        queue: asyncio.Queue[InvestigationEvent | None] = asyncio.Queue()
        self.queues[investigation_id].append(queue)
        try:
            while (event := await queue.get()) is not None:
                yield event
        finally:
            self.queues[investigation_id].remove(queue)

    async def report(self, investigation_id: str) -> dict | None:
        if investigation_id not in self.states:
            return None
        return self.reports.get(investigation_id, {"investigation_id": investigation_id, "status": "in_progress"})


def create_app(cache_dir=CACHE_DIR, documents_path=DATA_PATH) -> FastAPI:
    app = FastAPI(title="REASONING Investigation Engine", version="0.1.0")
    app.state.registry = InvestigationRegistry()
    app.state.investigator = Investigator(MockRetrievalClient(documents_path), CachedLLMClient(DiskLLMCache(cache_dir)))
    app.include_router(investigate.router)
    app.include_router(stream.router)
    app.include_router(timeline.router)
    app.include_router(compare.router)
    return app


app = create_app()
