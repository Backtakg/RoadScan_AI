# RoadScan model

Place the pothole-trained YOLO weights here as `best.pt` and start the API with:

```bash
ROADSCAN_MODEL=backend/models/best.pt uvicorn backend.main:app --reload --port 8000
```

Do not use the default COCO model for pothole claims. RoadScan requires a pothole-trained checkpoint. The repository currently keeps the model weights out of Git because large binary weights should not be committed until a specific model/dataset and its license are selected.

A candidate public dataset is the Roboflow `pothole_detection_version_2` dataset, which is listed as CC BY 4.0 and has a single `pothole` class. Verify the dataset terms and attribution requirements before using it for training or redistribution.
