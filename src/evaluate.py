"""Evaluate E0 or E2 YOLOv8s checkpoints on the same GTSDB validation split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = ROOT / "configs" / "gtsdb.yaml"
DEFAULT_WEIGHTS = ROOT / "runs" / "phase_b" / "weights" / "best.pt"
DEFAULT_OUTPUT = ROOT / "results" / "baseline"


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8s on GTSDB")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, ...")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if value != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu instead.")
    return value


def dataset_directory():
    with DATA_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    path = Path(config["path"])
    if not path.is_absolute():
        path = DATA_CONFIG.parent / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {path}")
    return path


def save_predictions(model, data_dir, split, device, count, imgsz, output_dir):
    if count <= 0:
        return
    image_dir = data_dir / "images" / split
    images = sorted(image_dir.glob("*.png"))
    if not images:
        raise RuntimeError(f"No PNG files in {image_dir}")

    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    random_state = np.random.default_rng(42)
    indices = random_state.choice(len(images), size=min(count, len(images)), replace=False)

    print(f"Saving {len(indices)} sample predictions to {predictions_dir}")
    for index in indices:
        image_path = images[int(index)]
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=0.25,
            iou=0.7,
            max_det=300,
            device=device,
            verbose=False,
        )
        results[0].save(filename=str(predictions_dir / image_path.name))


def main():
    args = parse_args()
    weights = project_path(args.weights)
    output_dir = project_path(args.output_dir)
    device = resolve_device(args.device)

    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {weights}")
    if not DATA_CONFIG.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_CONFIG}")
    if args.imgsz <= 0 or args.batch <= 0:
        raise ValueError("--imgsz and --batch must be positive")
    if args.samples < 0:
        raise ValueError("--samples must be nonnegative")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint: {weights}")
    print(f"Split: {args.split} | imgsz: {args.imgsz} | batch: {args.batch}")
    print(f"Output: {output_dir}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(DATA_CONFIG),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        conf=0.001,
        iou=0.7,
        max_det=300,
        plots=True,
        save_json=True,
        project=str(output_dir),
        name="validation" if args.split == "val" else "test_evaluation",
        exist_ok=True,
        verbose=True,
    )

    summary = {
        "model": "YOLOv8s",
        "weights": str(weights),
        "split": args.split,
        "image_size": args.imgsz,
        "batch_size": args.batch,
        "eval_conf": 0.001,
        "eval_nms_iou": 0.7,
        "max_detections": 300,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": {},
    }

    indices = [int(x) for x in metrics.box.ap_class_index]
    for class_id, class_name in model.names.items():
        class_id = int(class_id)
        if class_id not in indices:
            continue
        index = indices.index(class_id)
        summary["per_class"][str(class_id)] = {
            "name": str(class_name),
            "ap50_95": float(np.asarray(metrics.box.ap[index]).mean()),
            "ap50": float(metrics.box.ap50[index]),
        }

    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        summary["speed_ms_per_image"] = {
            str(key): float(value) for key, value in speed.items()
        }

    output_json = output_dir / "metrics.json"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFINAL METRICS")
    print(f"Precision:    {summary['precision']:.4f}")
    print(f"Recall:       {summary['recall']:.4f}")
    print(f"mAP@0.5:      {summary['map50']:.4f}")
    print(f"mAP@0.5:0.95: {summary['map50_95']:.4f}")
    print(f"Metrics JSON: {output_json}")

    save_predictions(
        model=model,
        data_dir=dataset_directory(),
        split=args.split,
        device=device,
        count=args.samples,
        imgsz=args.imgsz,
        output_dir=output_dir,
    )
    print(f"Evaluation completed: {output_dir}")


if __name__ == "__main__":
    main()
