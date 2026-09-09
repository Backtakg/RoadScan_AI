from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

DB_PATH = os.getenv("ROADSCAN_DB", "backend/data/roadscan.sqlite3")
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

class HistoryStore:
    def __init__(self, path: str = DB_PATH) -> None:
        self.path = path
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS inspections (
                id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT NOT NULL,
                frames INTEGER NOT NULL DEFAULT 0, potholes INTEGER NOT NULL DEFAULT 0,
                high INTEGER NOT NULL DEFAULT 0, medium INTEGER NOT NULL DEFAULT 0,
                low INTEGER NOT NULL DEFAULT 0, route_points INTEGER NOT NULL DEFAULT 0,
                route_distance_m REAL NOT NULL DEFAULT 0, analytics_json TEXT NOT NULL,
                events_json TEXT NOT NULL, route_json TEXT NOT NULL
            )""")
            conn.commit()

    def save(self, started_at: str, ended_at: str, frames: int, summary: dict[str, Any], analytics: dict[str, Any], events: list[dict[str, Any]], route: dict[str, Any]) -> str:
        inspection_id = f"INS-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        with self._connect() as conn:
            conn.execute("INSERT INTO inspections VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (inspection_id, started_at, ended_at, frames, summary.get("total", 0), summary.get("high", 0), summary.get("medium", 0), summary.get("low", 0), route.get("pointCount", 0), route.get("distanceMeters", 0), json.dumps(analytics), json.dumps(events), json.dumps(route)))
            conn.commit()
        return inspection_id

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id,started_at,ended_at,frames,potholes,high,medium,low,route_points,route_distance_m,analytics_json FROM inspections ORDER BY started_at DESC LIMIT ?", (max(1, min(limit, 100)),)).fetchall()
        output = []
        for row in rows:
            item = dict(row); item["analytics"] = json.loads(item.pop("analytics_json")); output.append(item)
        return output

    def trends(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.list(limit)
        return [{"id": r["id"], "startedAt": r["started_at"], "potholes": r["potholes"], "routeDistanceKm": round(r["route_distance_m"] / 1000, 3), "potholesPerKm": r["analytics"].get("potholesPerKm"), "conditionScore": r["analytics"].get("conditionScore"), "conditionLabel": r["analytics"].get("conditionLabel"), "confidencePercent": r["analytics"].get("confidencePercent", 0)} for r in reversed(rows)]

    def get(self, inspection_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM inspections WHERE id = ?", (inspection_id,)).fetchone()
        if row is None: return None
        item = dict(row); item["analytics"] = json.loads(item.pop("analytics_json")); item["events"] = json.loads(item.pop("events_json")); item["route"] = json.loads(item.pop("route_json")); return item

    def delete(self, inspection_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM inspections WHERE id = ?", (inspection_id,)); conn.commit(); return cursor.rowcount > 0

history = HistoryStore()
