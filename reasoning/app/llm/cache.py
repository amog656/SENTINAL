from __future__ import annotations

import hashlib
import json
from pathlib import Path


class LLMCacheMiss(RuntimeError):
    pass


class DiskLLMCache:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, prompt: str, model: str, temperature: float) -> Path:
        key = hashlib.sha256(f"{prompt}\0{model}\0{temperature}".encode()).hexdigest()
        return self.directory / f"{key}.json"

    def get(self, prompt: str, model: str, temperature: float) -> str | None:
        path = self._path(prompt, model, temperature)
        return json.loads(path.read_text())["response"] if path.exists() else None

    def put(self, prompt: str, model: str, temperature: float, response: str) -> None:
        self._path(prompt, model, temperature).write_text(json.dumps({"response": response}))
