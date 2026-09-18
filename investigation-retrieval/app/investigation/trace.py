from typing import Any

from app.investigation.models import TraceEvent


class InvestigationTrace:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(self, event: str, hop: int, **details: Any) -> None:
        self.events.append(TraceEvent(event=event, hop=hop, details=details))

    def as_list(self) -> list[TraceEvent]:
        return self.events
