from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from math import hypot
from typing import Any


@dataclass
class PotholeEvent:
    event_id: str
    track_id: int
    timestamp: str
    latitude: float | None
    longitude: float | None
    confidence: float
    bbox: list[float]
    severity: str
    evidence: str = "camera_frame"


class InspectionStore:
    """In-memory event store for the prototype. Replace with SQLite/Postgres later."""

    def __init__(self) -> None:
        self.events: dict[int, PotholeEvent] = {}
        self.sequence = 0

    def upsert(self, track_id: int, detection: dict[str, Any], latitude: float | None, longitude: float | None, timestamp: str | None) -> PotholeEvent:
        now = timestamp or datetime.now(timezone.utc).isoformat()
        if track_id in self.events:
            event = self.events[track_id]
            event.confidence = max(event.confidence, float(detection["confidence"]))
            event.bbox = detection["bbox"]
            event.timestamp = now
            event.latitude = latitude
            event.longitude = longitude
            return event
        self.sequence += 1
        confidence = float(detection["confidence"])
        severity = "high" if confidence >= 0.75 else "medium" if confidence >= 0.5 else "low"
        event = PotholeEvent(f"PTH-{self.sequence:04d}", track_id, now, latitude, longitude, confidence, detection["bbox"], severity)
        self.events[track_id] = event
        return event

    def summary(self) -> dict[str, int]:
        values = list(self.events.values())
        return {"total": len(values), "high": sum(e.severity == "high" for e in values), "medium": sum(e.severity == "medium" for e in values), "low": sum(e.severity == "low" for e in values)}

    def list_events(self) -> list[dict[str, Any]]:
        return [asdict(event) for event in self.events.values()]


store = InspectionStore()
