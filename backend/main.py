from __future__ import annotations

import base64
import os
import time
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from ultralytics import YOLO

from inspection_store import store
from report_generator import build_inspection_pdf

MODEL_PATH = os.getenv("ROADSCAN_MODEL", "backend/models/best.pt")
TRACKER_PATH = os.getenv("ROADSCAN_TRACKER", "backend/trackers/roadscan_bytetrack.yaml")
CONFIDENCE = float(os.getenv("ROADSCAN_CONFIDENCE", "0.35"))
EVIDENCE_DIR = os.getenv("ROADSCAN_EVIDENCE_DIR", "backend/data/evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

app = FastAPI(title="RoadScan AI Vision API", version="0.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/evidence", StaticFiles(directory=EVIDENCE_DIR), name="evidence")
_model: YOLO | None = None


class FrameRequest(BaseModel):
    image: str = Field(description="Base64 JPEG/PNG data, optionally prefixed with a data URL")
    latitude: float | None = None
    longitude: float | None = None
    timestamp: str | None = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Pothole model not found at {MODEL_PATH}. Place a trained pothole YOLO weights file there or set ROADSCAN_MODEL.",
            )
        _model = YOLO(MODEL_PATH)
    return _model


def decode_image(value: str) -> np.ndarray:
    try:
        payload = value.split(",", 1)[1] if "," in value else value
        raw = base64.b64decode(payload)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to decode image")
        return image
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc


def save_evidence(image: np.ndarray, event_id: str) -> str:
    filename = f"{event_id}.jpg"
    path = os.path.join(EVIDENCE_DIR, filename)
    if not cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise RuntimeError(f"Unable to save evidence frame to {path}")
    return f"/evidence/{filename}"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_loaded": _model is not None,
        "tracker": TRACKER_PATH,
        "tracker_exists": os.path.exists(TRACKER_PATH),
        "evidence_directory": EVIDENCE_DIR,
        "route_points": store.route_data()["pointCount"],
    }


@app.post("/detect")
def detect(request: FrameRequest) -> dict[str, Any]:
    image = decode_image(request.image)
    started = time.perf_counter()
    model = get_model()
    store.add_route_point(request.latitude, request.longitude, request.timestamp)
    results = model.track(image, conf=CONFIDENCE, tracker=TRACKER_PATH, persist=True, verbose=False)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    detections: list[dict[str, Any]] = []
    new_events: list[dict[str, Any]] = []
    result = results[0]
    names = result.names
    boxes = result.boxes
    track_ids = boxes.id.int().tolist() if boxes is not None and boxes.id is not None else []

    if boxes is not None:
        for index, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())
            bbox = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            track_id = int(track_ids[index]) if index < len(track_ids) else None
            detections.append({
                "classId": cls_id,
                "label": names.get(cls_id, str(cls_id)),
                "confidence": round(confidence, 3),
                "bbox": bbox,
                "trackId": track_id,
            })

            if track_id is not None:
                event, is_new = store.upsert(
                    track_id=track_id,
                    detection={"confidence": confidence, "bbox": bbox},
                    latitude=request.latitude,
                    longitude=request.longitude,
                    timestamp=request.timestamp,
                )
                if is_new:
                    try:
                        evidence_path = save_evidence(image, event.event_id)
                        store.set_evidence(event.event_id, evidence_path)
                    except Exception as exc:
                        raise HTTPException(status_code=500, detail=f"Evidence capture failed: {exc}") from exc
                    new_events.append(event.to_dict())

    return {
        "detections": detections,
        "events": store.list_events(),
        "newEvents": new_events,
        "summary": store.summary(),
        "route": store.route_data(),
        "inferenceMs": elapsed_ms,
        "timestamp": request.timestamp,
        "gps": {"latitude": request.latitude, "longitude": request.longitude},
        "source": "processed_camera_frame",
        "model": MODEL_PATH,
        "tracker": TRACKER_PATH,
    }


@app.get("/events")
def events() -> dict[str, Any]:
    return {"events": store.list_events(), "summary": store.summary(), "route": store.route_data()}


@app.get("/route")
def route() -> dict[str, Any]:
    return store.route_data()


@app.get("/summary")
def summary() -> dict[str, Any]:
    return store.summary()


@app.get("/report.pdf", response_class=Response)
def report_pdf() -> Response:
    """Generate a PDF from the current in-memory inspection session."""
    try:
        pdf = build_inspection_pdf(store.list_events(), store.route_data(), EVIDENCE_DIR)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}") from exc
    filename = f"roadscan-inspection-{time.strftime('%Y%m%d-%H%M%S')}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/reset")
def reset() -> dict[str, Any]:
    global _model
    store.reset()
    _model = None
    return {"status": "ok", "summary": store.summary(), "route": store.route_data()}
