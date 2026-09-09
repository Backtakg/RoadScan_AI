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

    def upsert(self, track_id: int, detection: dict[str, Any], latitude: float | None, longitude: float | None, timestamp: str | None) -> tuple[PotholeEvent, bool]:
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
        # AI confidence band, not engineering severity.
        severity = "high" if confidence >= 0.75 else "medium" if confidence >= 0.5 else "low"
        event = PotholeEvent(f"PTH-{self.sequence:04d}", track_id, now, latitude, longitude, confidence, detection["bbox"], severity)
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
        point = {"latitude": float(latitude), "longitude": float(longitude), "timestamp": timestamp or datetime.now(timezone.utc).isoformat()}
        if self.route:
            previous = self.route[-1]
            if _distance_m(previous["latitude"], previous["longitude"], point["latitude"], point["longitude"]) < 2.0:
                return
        self.route.append(point)

    def route_data(self) -> dict[str, Any]:
        distance_m = sum(_distance_m(a["latitude"], a["longitude"], b["latitude"], b["longitude"]) for a, b in zip(self.route, self.route[1:]))
        return {"points": self.route, "distanceMeters": round(distance_m, 1), "distanceKm": round(distance_m / 1000, 3), "pointCount": len(self.route)}

    def summary(self) -> dict[str, int | float]:
        values = list(self.events.values())
        route = self.route_data()
        return {"total": len(values), "high": sum(e.severity == "high" for e in values), "medium": sum(e.severity == "medium" for e in values), "low": sum(e.severity == "low" for e in values), "routePoints": route["pointCount"], "routeDistanceMeters": route["distanceMeters"]}

    def analytics(self) -> dict[str, Any]:
        values = list(self.events.values())
        route = self.route_data()
        distance_km = float(route["distanceKm"])
        total = len(values)
        mapped = [e for e in values if e.latitude is not None and e.longitude is not None]
        avg_conf = sum(e.confidence for e in values) / total if total else 0.0
        density = total / distance_km if distance_km > 0 else None
        score = max(0.0, min(100.0, 100.0 - (density * 10.0 if density is not None else 0.0)))
        if total == 0:
            score = None
        hotspots = []
        for event in mapped:
            hotspots.append({"eventId": event.event_id, "latitude": event.latitude, "longitude": event.longitude, "confidence": round(event.confidence, 3), "band": event.severity})
        return {"potholesPerKm": round(density, 2) if density is not None else None, "averageConfidence": round(avg_conf, 3), "confidencePercent": round(avg_conf * 100, 1), "conditionScore": round(score, 1) if score is not None else None, "conditionLabel": _condition_label(score), "mappedPotholes": len(mapped), "unmappedPotholes": total - len(mapped), "evidenceCaptured": sum(bool(e.evidence) for e in values), "hotspots": hotspots}

    def list_events(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events.values()]

    def reset(self) -> None:
        self.events.clear()
        self.sequence = 0
        self.route.clear()


def _condition_label(score: float | None) -> str:
    if score is None:
        return "No data"
    if score >= 80:
        return "Good"
    if score >= 60:
        return "Watch"
    if score >= 40:
        return "Poor"
    return "Critical"


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius = 6_371_000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi, dlambda = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return earth_radius * 2 * asin(sqrt(max(0.0, min(1.0, a))))


store = InspectionStore()
