from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InspectionStore:
    """In-memory inspection events. Replace with SQLite/Postgres for persistence."""

    def __init__(self) -> None:
        self.events: dict[int, PotholeEvent] = {}
        self.sequence = 0

    def upsert(
        self,
        track_id: int,
        detection: dict[str, Any],
        latitude: float | None,
        longitude: float | None,
        timestamp: str | None,
    ) -> tuple[PotholeEvent, bool]:
        now = timestamp or datetime.now(timezone.utc).isoformat()
        if track_id in self.events:
            event = self.events[track_id]
            event.confidence = max(event.confidence, float(detection["confidence"]))
            event.bbox = detection["bbox"]
            event.timestamp = now
            event.latitude = latitude
            event.longitude = longitude
            return event, False

        self.sequence += 1
        confidence = float(detection["confidence"])
        # This is an AI confidence band, not an engineering assessment of pothole severity.
        severity = "high" if confidence >= 0.75 else "medium" if confidence >= 0.5 else "low"
        event = PotholeEvent(
            event_id=f"PTH-{self.sequence:04d}",
            track_id=track_id,
            timestamp=now,
            latitude=latitude,
            longitude=longitude,
            confidence=confidence,
            bbox=detection["bbox"],
            severity=severity,
        )
        self.events[track_id] = event
        return event, True

    def set_evidence(self, event_id: str, evidence: str) -> PotholeEvent | None:
        for event in self.events.values():
            if event.event_id == event_id:
                event.evidence = evidence
                return event
        return None

    def summary(self) -> dict[str, int]:
        values = list(self.events.values())
        return {
            "total": len(values),
            "high": sum(e.severity == "high" for e in values),
            "medium": sum(e.severity == "medium" for e in values),
            "low": sum(e.severity == "low" for e in values),
        }

    def list_events(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events.values()]

    def reset(self) -> None:
        self.events.clear()
        self.sequence = 0


store = InspectionStore()
