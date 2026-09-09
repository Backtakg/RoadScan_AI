# RoadScan AI — All-Weather AI Pipeline

RoadScan's production API can run one YOLO model, but detection quality in night, rain, fog and clear daylight must be measured from real labelled data. This directory provides the reproducible pipeline for doing that without claiming unverified accuracy.

## Target conditions

Every evaluation set should contain real images/video frames labelled for potholes and tagged with one of:

- `day_clear`
- `night`
- `rain`
- `fog`
- `mixed` when a frame contains more than one difficult condition

Do not put synthetic benchmark images into the held-out test set. Synthetic augmentation is for training robustness, not for reporting real-world accuracy.

## Training

1. Collect representative road images from the deployment camera, including Nepal roads where possible.
2. Label potholes with YOLO bounding boxes.
3. Split by **drive/route**, not by adjacent frames, so near-identical frames cannot leak between train and validation/test.
4. Train the same model configuration on the training set with moderate photometric and weather augmentation.
5. Keep a completely untouched condition-specific test set.
6. Copy only the validated production weights to `backend/models/best.pt`.

Example Ultralytics training command:

```bash
yolo detect train model=yolo11n.pt data=dataset.yaml epochs=100 imgsz=640 batch=16 project=runs name=roadscan_all_weather
```

Use the model version supported by the installed Ultralytics package. The command above is an example; it is **not** a claim that a pretrained model is already accurate for potholes.

## Condition-specific benchmark

Run:

```bash
python ml/benchmark.py --model runs/roadscan_all_weather/weights/best.pt --data dataset.yaml --out ml/benchmark-results.json
```

The benchmark evaluates each condition separately and reports precision, recall and mAP50/mAP50-95 from the real held-out set. It also reports sample counts. No accuracy number is hard-coded anywhere in the application.

## Deployment gate

Do not deploy a new model merely because its overall score is higher. Compare:

- clear daylight
- night
- rain
- fog
- mixed conditions

A model is considered a candidate only when it improves or preserves the required condition-specific metrics and has no unacceptable regression on clear-road performance.

## Runtime behavior

The application receives a model confidence value from YOLO. It maps that value to an **AI confidence band**:

- High: >= 0.75
- Medium: >= 0.50 and < 0.75
- Low: < 0.50

These bands are not engineering pothole severity.

The frontend should also display a difficult-condition warning when the camera reports low-light/poor visibility. That warning is a limitation notice, not a fabricated weather classifier.
