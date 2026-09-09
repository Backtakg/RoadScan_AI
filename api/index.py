"""RoadScan AI Vercel API.

Uses ONNX Runtime instead of PyTorch/Ultralytics so the detector can run inside
Vercel's Python runtime without the multi-gigabyte dependency bundle.
The model is downloaded lazily from a public Apache-2.0 Hugging Face model.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

MODEL_URL = os.getenv(
    "ROADSCAN_MODEL_URL",
    "https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.onnx",
)
MODEL_PATH = Path("/tmp/roadscan-pothole.onnx")
CONFIDENCE = float(os.getenv("ROADSCAN_CONFIDENCE", "0.35"))
IOU_THRESHOLD = float(os.getenv("ROADSCAN_IOU", "0.45"))

app = FastAPI(title="RoadScan AI Vision API", version="2.0.0-onnx")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_session: ort.InferenceSession | None = None
_inspection: dict[str, Any] | None = None


class FrameRequest(BaseModel):
    image: str
    latitude: float | None = None
    longitude: float | None = None
    timestamp: str | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_session() -> ort.InferenceSession:
    global _session
    if _session is not None:
        return _session
    if not MODEL_PATH.exists():
        tmp = MODEL_PATH.with_suffix(".download")
        try:
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.replace(MODEL_PATH)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise HTTPException(status_code=503, detail=f"Unable to download pothole model: {exc}")
    try:
        _session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Unable to load ONNX pothole model: {exc}")
    return _session


def decode_image(data: str) -> np.ndarray:
    if "," in data and data.startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}")
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return image


def preprocess(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas_img = np.full((size, size, 3), 114, dtype=np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas_img[dy : dy + nh, dx : dx + nw] = resized
    rgb = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(tensor, 0), scale, (dx, dy)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        remaining = np.where(inter / union <= threshold)[0]
        order = order[remaining + 1]
    return keep


def infer(image: np.ndarray) -> list[dict[str, Any]]:
    session = load_session()
    tensor, scale, (dx, dy) = preprocess(image)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]
    pred = np.asarray(output)
    if pred.ndim == 3:
        pred = pred[0]
    # Ultralytics YOLOv8 ONNX export is normally [4 + classes, anchors].
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    if pred.shape[1] < 5:
        return []
    boxes_xywh = pred[:, :4]
    scores = pred[:, 4:].max(axis=1)
    mask = scores >= CONFIDENCE
    boxes_xywh, scores = boxes_xywh[mask], scores[mask]
    if len(scores) == 0:
        return []
    cx, cy, bw, bh = boxes_xywh.T
    x1 = (cx - bw / 2 - dx) / scale
    y1 = (cy - bh / 2 - dy) / scale
    x2 = (cx + bw / 2 - dx) / scale
    y2 = (cy + bh / 2 - dy) / scale
    h, w = image.shape[:2]
    boxes = np.column_stack((
        np.clip(x1, 0, w), np.clip(y1, 0, h),
        np.clip(x2, 0, w), np.clip(y2, 0, h),
    )).astype(np.float32)
    keep = nms(boxes, scores, IOU_THRESHOLD)
    return [
        {
            "label": "pothole",
            "confidence": float(scores[i]),
            "bbox": [float(v) for v in boxes[i]],
        }
        for i in keep
    ]


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + ab - inter + 1e-6)


def track_detections(detections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert _inspection is not None
    tracks = _inspection.setdefault("tracks", {})
    next_id = int(_inspection.setdefault("next_track_id", 1))
    events = _inspection.setdefault("events", [])
    used: set[int] = set()
    current: list[dict[str, Any]] = []
    for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        best_id = None
        best_iou = 0.0
        for tid, track in tracks.items():
            tid_int = int(tid)
            if tid_int in used:
                continue
            score = iou(det["bbox"], track["bbox"])
            if score > best_iou:
                best_iou, best_id = score, tid_int
        if best_id is None or best_iou < 0.25:
            best_id = next_id
            next_id += 1
            tracks[str(best_id)] = {"bbox": det["bbox"], "age": 0, "hits": 1}
            is_new = True
        else:
            tracks[str(best_id)]["bbox"] = det["bbox"]
            tracks[str(best_id)]["age"] = 0
            tracks[str(best_id)]["hits"] = int(tracks[str(best_id)].get("hits", 0)) + 1
            is_new = False
        used.add(best_id)
        det = {**det, "trackId": best_id}
        current.append(det)
        if is_new:
            event = {
                "event_id": str(uuid.uuid4()),
                "track_id": best_id,
                "timestamp": now_iso(),
                "latitude": _inspection.get("latitude"),
                "longitude": _inspection.get("longitude"),
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "severity": "high" if det["confidence"] >= 0.75 else "medium" if det["confidence"] >= 0.5 else "low",
                "evidence": "",
            }
            events.append(event)
    for track in tracks.values():
        track["age"] = int(track.get("age", 0)) + 1
    _inspection["next_track_id"] = next_id
    return current, events


def summary() -> dict[str, Any]:
    events = (_inspection or {}).get("events", [])
    counts = {"high": 0, "medium": 0, "low": 0}
    for e in events:
        counts[e["severity"]] += 1
    route = (_inspection or {}).get("route", [])
    return {
        "total": len(events), **counts,
        "routePoints": len(route), "routeDistanceMeters": route_distance(route),
    }


def route_distance(points: list[dict[str, Any]]) -> float:
    total = 0.0
    for a, b in zip(points, points[1:]):
        lat1, lon1, lat2, lon2 = map(float, [a["latitude"], a["longitude"], b["latitude"], b["longitude"]])
        r = 6371000.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
        h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        total += 2 * r * np.arcsin(np.sqrt(h))
    return float(total)


def analytics() -> dict[str, Any]:
    s = summary()
    distance = float(s["routeDistanceMeters"])
    total = int(s["total"])
    per_km = total / (distance / 1000.0) if distance > 1 else None
    avg = float(np.mean([e["confidence"] for e in (_inspection or {}).get("events", [])])) if total else 0.0
    score = max(0, min(100, round(100 - (per_km or 0) * 8))) if per_km is not None else None
    return {
        "potholesPerKm": round(per_km, 2) if per_km is not None else None,
        "averageConfidence": avg,
        "confidencePercent": avg * 100,
        "conditionScore": score,
        "conditionLabel": "Good" if score is not None and score >= 80 else "Watch" if score is not None and score >= 60 else "Needs review" if score is not None else "No data",
        "mappedPotholes": sum(1 for e in (_inspection or {}).get("events", []) if e.get("latitude") is not None),
        "unmappedPotholes": sum(1 for e in (_inspection or {}).get("events", []) if e.get("latitude") is None),
        "evidenceCaptured": sum(1 for e in (_inspection or {}).get("events", []) if e.get("evidence")),
        "hotspots": [
            {"eventId": e["event_id"], "latitude": e["latitude"], "longitude": e["longitude"], "confidence": e["confidence"], "band": e["severity"]}
            for e in (_inspection or {}).get("events", [])
        ],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "roadscan-vision", "model": "YOLOv8 pothole ONNX", "modelLoaded": _session is not None}


@app.post("/history/start")
def history_start() -> dict[str, Any]:
    global _inspection
    _inspection = {"id": str(uuid.uuid4()), "started_at": now_iso(), "frames": 0, "events": [], "route": [], "tracks": {}, "next_track_id": 1, "latitude": None, "longitude": None}
    return {"id": _inspection["id"], "startedAt": _inspection["started_at"]}


@app.post("/detect")
def detect(request: FrameRequest) -> dict[str, Any]:
    if _inspection is None:
        history_start()
    assert _inspection is not None
    image = decode_image(request.image)
    started = time.perf_counter()
    _inspection["latitude"], _inspection["longitude"] = request.latitude, request.longitude
    if request.latitude is not None and request.longitude is not None:
        _inspection["route"].append({"latitude": request.latitude, "longitude": request.longitude, "timestamp": request.timestamp or now_iso()})
    detections = infer(image)
    current, events = track_detections(detections)
    _inspection["frames"] += 1
    latency = round((time.perf_counter() - started) * 1000, 1)
    return {
        "detections": current,
        "events": events,
        "summary": summary(),
        "analytics": analytics(),
        "route": {"points": _inspection["route"], "distanceMeters": route_distance(_inspection["route"]), "distanceKm": route_distance(_inspection["route"]) / 1000, "pointCount": len(_inspection["route"])},
        "frames": _inspection["frames"],
        "inferenceMs": latency,
        "gps": {"latitude": request.latitude, "longitude": request.longitude},
        "source": "real_camera_frame",
        "model": "peterhdd/pothole-detection-yolov8 ONNX",
        "tracker": "RoadScan IoU tracker",
    }


@app.post("/history/complete")
def history_complete() -> dict[str, Any]:
    if _inspection is None:
        raise HTTPException(status_code=400, detail="No active inspection")
    _inspection["ended_at"] = now_iso()
    return {"id": _inspection["id"], "summary": summary(), "analytics": analytics()}


@app.post("/reset")
def reset() -> dict[str, Any]:
    global _inspection
    _inspection = None
    return {"ok": True}


@app.get("/events")
def events() -> list[dict[str, Any]]:
    return (_inspection or {}).get("events", [])


@app.get("/route")
def route() -> dict[str, Any]:
    points = (_inspection or {}).get("route", [])
    d = route_distance(points)
    return {"points": points, "distanceMeters": d, "distanceKm": d / 1000, "pointCount": len(points)}


@app.get("/summary")
def get_summary() -> dict[str, Any]:
    return summary()


@app.get("/analytics")
def get_analytics() -> dict[str, Any]:
    return analytics()


@app.get("/maintenance")
def maintenance() -> dict[str, Any]:
    return {"segments": [], "repairQueue": [], "recurring": []}


@app.get("/history")
def history() -> list[dict[str, Any]]:
    if _inspection is None:
        return []
    return [{"id": _inspection["id"], "started_at": _inspection["started_at"], "ended_at": _inspection.get("ended_at"), "frames": _inspection["frames"], "potholes": len(_inspection["events"]), **summary()}]


@app.get("/history/trends")
def trends() -> dict[str, Any]:
    return {"trends": history()}


@app.get("/history/compare")
def compare() -> dict[str, Any]:
    return {"inspections": history()}


@app.get("/report.pdf")
def report_pdf() -> Response:
    if _inspection is None:
        raise HTTPException(status_code=400, detail="No inspection available")
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(50, height - 60, "RoadScan AI — Inspection Report")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, height - 82, f"Inspection: {_inspection['id']}")
    pdf.drawString(50, height - 98, f"Started: {_inspection['started_at']}")
    y = height - 135
    s = summary()
    for label, value in [("Unique potholes", s["total"]), ("High confidence", s["high"]), ("Medium confidence", s["medium"]), ("Low confidence", s["low"]), ("Frames processed", _inspection["frames"]), ("Route points", len(_inspection["route"])), ("Route distance (m)", round(s["routeDistanceMeters"], 1))]:
        pdf.drawString(60, y, f"{label}: {value}")
        y -= 18
    pdf.drawString(60, y - 5, "Note: confidence bands are AI confidence, not engineering severity.")
    pdf.showPage()
    pdf.save()
    return Response(content=buf.getvalue(), media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=roadscan-inspection-report.pdf"})
