from __future__ import annotations

import base64
import os
import time
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ultralytics import YOLO

from inspection_store import store

MODEL_PATH = os.getenv("ROADSCAN_MODEL", "backend/models/best.pt")
TRACKER_PATH = os.getenv("ROADSCAN_TRACKER", "backend/trackers/roadscan_bytetrack.yaml")
CONFIDENCE = float(os.getenv("ROADSCAN_CONFIDENCE", "0.35"))

app = FastAPI(title="RoadScan AI Vision API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
                detail=(
                    f"Pothole model not found at {MODEL_PATH}. "
                    "Place a trained pothole YOLO weights file there or set ROADSCAN_MODEL."
                ),
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


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_loaded": _model is not None,
        "tracker": TRACKER_PATH,
        "tracker_exists": os.path.exists(TRACKER_PATH),
    }


@app.post("/detect")
def detect(request: FrameRequest) -> dict[str, Any]:
    image = decode_image(request.image)
    started = time.perf_counter()
    model = get_model()

    # Frames from the same live camera stream are passed consecutively with
    # persist=True so ByteTrack can maintain IDs across frames.
    results = model.track(
        image,
        conf=CONFIDENCE,
        tracker=TRACKER_PATH,
        persist=True,
        verbose=False,
    )
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
            x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            track_id = int(track_ids[index]) if index < len(track_ids) else None

            detection = {
                "classId": cls_id,
                "label": names.get(cls_id, str(cls_id)),
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2],
                "trackId": track_id,
            }
            detections.append(detection)

            if track_id is not None:
                event = store.upsert(
                    track_id=track_id,
                    timestamp=request.timestamp,
                    latitude=request.latitude,
                    longitude=request.longitude,
                    confidence=confidence,
                    bbox=[x1, y1, x2, y2],
                )
                event_dict = event.to_dict()
                if event.is_new:
                    new_events.append(event_dict)

    return {
        "detections": detections,
        "newEvents": new_events,
        "summary": store.summary(),
        "inferenceMs": elapsed_ms,
        "timestamp": request.timestamp,
        "gps": {"latitude": request.latitude, "longitude": request.longitude},
        "source": "processed_camera_frame",
        "model": MODEL_PATH,
        "tracker": TRACKER_PATH,
    }


@app.get("/events")
def events() -> dict[str, Any]:
    return {"events": [event.to_dict() for event in store.list_events()], "summary": store.summary()}


@app.get("/summary")
def summary() -> dict[str, Any]:
    return store.summary()


@app.post("/reset")
def reset() -> dict[str, Any]:
    store.reset()
    return {"status": "ok", "summary": store.summary()}
