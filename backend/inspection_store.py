from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt
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
    """In-memory inspection events and sampled GPS route."""

    def __init__(self) -> None:
        self.events: dict[int, PotholeEvent] = {}
        self.sequence = 0
        self.route: list[dict[str, Any]] = []

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

    def add_route_point(self, latitude: float | None, longitude: float | None, timestamp: str | None) -> None:
        if latitude is None or longitude is None:
            return
        point = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
        if self.route:
            previous = self.route[-1]
            if _distance_m(previous["latitude"], previous["longitude"], point["latitude"], point["longitude"]) < 2.0:
                return
        self.route.append(point)

    def route_data(self) -> dict[str, Any]:
        distance_m = 0.0
        for previous, current in zip(self.route, self.route[1:]):
            distance_m += _distance_m(previous["latitude"], previous["longitude"], current["latitude"], current["longitude"])
        return {
            "points": self.route,
            "distanceMeters": round(distance_m, 1),
            "distanceKm": round(distance_m / 1000, 3),
            "pointCount": len(self.route),
        }

    def summary(self) -> dict[str, int | float]:
        values = list(self.events.values())
        route = self.route_data()
        return {
            "total": len(values),
            "high": sum(e.severity == "high" for e in values),
            "medium": sum(e.severity == "medium" for e in values),
            "low": sum(e.severity == "low" for e in values),
            "routePoints": route["pointCount"],
            "routeDistanceMeters": route["distanceMeters"],
        }

    def list_events(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events.values()]

    def reset(self) -> None:
        self.events.clear()
        self.sequence = 0
        self.route.clear()


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius = 6_371_000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return earth_radius * 2 * asin(sqrt(max(0.0, min(1.0, a))))


store = InspectionStore()
