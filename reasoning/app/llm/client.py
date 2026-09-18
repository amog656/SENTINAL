from __future__ import annotations

import asyncio
import os
from collections.abc import Callable

from app.llm.cache import DiskLLMCache, LLMCacheMiss


class CachedLLMClient:
    """Cache wrapper. Wire a real provider into `_network_complete` in Phase 4."""

    def __init__(self, cache: DiskLLMCache, model: str = "demo-deterministic-phase2"):
        self.cache, self.model, self.temperature = cache, model, 0.0

    async def complete(self, prompt: str, fallback: Callable[[], str]) -> str:
        cached = self.cache.get(prompt, self.model, self.temperature)
        if cached is not None:
            return cached
        if os.getenv("DEMO_MODE", "").lower() == "replay":
            raise LLMCacheMiss("DEMO_MODE=replay cache miss; record this demo question before presenting.")
        try:
            # Current MVP uses a deterministic local fallback. A real client goes here later.
            response = await asyncio.wait_for(asyncio.to_thread(fallback), timeout=10)
        except Exception:
            # Exactly one retry; callers keep investigating if this still fails.
            response = await asyncio.wait_for(asyncio.to_thread(fallback), timeout=10)
        self.cache.put(prompt, self.model, self.temperature, response)
        return response
