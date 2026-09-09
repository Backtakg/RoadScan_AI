"""Condition-specific YOLO benchmark for RoadScan AI.

The dataset YAML must contain test images/labels and, for condition-aware
reporting, a JSON manifest mapping each test image to a real condition.

Manifest format:
{
  "images/road001.jpg": "day_clear",
  "images/road002.jpg": "night",
  "images/road003.jpg": "rain",
  "images/road004.jpg": "fog"
}

This script deliberately produces measured metrics only. It never invents
accuracy values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--manifest", required=False, help="JSON image -> condition mapping")
    parser.add_argument("--out", default="ml/benchmark-results.json")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install the backend ML dependencies before benchmarking: pip install -r backend/requirements.txt") from exc

    model = YOLO(args.model)
    base = model.val(data=args.data, split="test", verbose=False)

    result = {
        "model": str(args.model),
        "dataset": str(args.data),
        "overall": {
            "precision": float(base.box.mp),
            "recall": float(base.box.mr),
            "map50": float(base.box.map50),
            "map50_95": float(base.box.map),
        },
        "conditions": {},
        "note": "Metrics are measured on the supplied dataset; condition metrics require a manifest and are never inferred.",
    }

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise SystemExit("Manifest must be a JSON object mapping image paths to condition names.")

        # Per-condition metrics are evaluated through temporary dataset YAMLs.
        # This keeps the benchmark honest: each condition uses only its labelled
        # test images rather than guessing weather from pixels.
        import tempfile
        import yaml

        original = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
        test_root = Path(original.get("path", "."))
        test_images = original.get("test")
        if isinstance(test_images, list):
            image_root = test_root
        else:
            image_root = test_root / str(test_images)

        for condition in sorted(set(manifest.values())):
            selected = []
            for image, tag in manifest.items():
                if tag != condition:
                    continue
                selected.append(str(image))
            if not selected:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                list_file = Path(tmp) / f"{condition}.txt"
                list_file.write_text("\n".join(selected) + "\n", encoding="utf-8")
                condition_yaml = dict(original)
                condition_yaml["test"] = str(list_file)
                condition_yaml.pop("path", None)
                condition_yaml["path"] = str(test_root)
                data_file = Path(tmp) / "data.yaml"
                data_file.write_text(yaml.safe_dump(condition_yaml), encoding="utf-8")
                metrics = model.val(data=str(data_file), split="test", verbose=False)
                result["conditions"][condition] = {
                    "images": len(selected),
                    "precision": float(metrics.box.mp),
                    "recall": float(metrics.box.mr),
                    "map50": float(metrics.box.map50),
                    "map50_95": float(metrics.box.map),
                }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
