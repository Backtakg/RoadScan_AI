# RoadScan AI Vision API

Local FastAPI inference service for the RoadScan AI web app.

## Run locally

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The first inference downloads the configured Ultralytics model if it is not already present. Set `ROADSCAN_MODEL` to a pothole-trained YOLO weights file when available.

## Endpoint

- `GET /health` — service status
- `POST /detect` — accepts a base64 camera frame plus optional GPS/timestamp and returns bounding boxes, confidence and inference time.

**Important:** the default YOLO model is a general object detector and is not a pothole detector. For real pothole results, provide a pothole-trained model via `ROADSCAN_MODEL`.
