
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "runs" / "phase_b" / "weights" / "best.pt"
DATA_CONFIG = ROOT / "configs" / "gtsdb.yaml"
RESULTS_DIR = ROOT / "results" / "baseline"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8s on GTSDB"
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of prediction images to save",
    )

    return parser.parse_args()


def save_predictions(model, dataset_dir, split, device, count):
    image_dir = dataset_dir / "images" / split
    output_dir = RESULTS_DIR / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob("*.png"))

    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    rng = np.random.default_rng(42)
    selected = rng.choice(
        len(images),
        size=min(count, len(images)),
        replace=False,
    )

    print(f"\nSaving predictions for {len(selected)} images...")

    for index in selected:
        image = images[int(index)]

        results = model.predict(
            source=str(image),
            imgsz=640,
            conf=0.25,
            device=device,
            verbose=False,
        )

        output = output_dir / image.name
        results[0].save(filename=str(output))

        print(f"Saved: {output.name}")


def main():
    args = parse_args()

    device = (
        0 if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device

    weights = args.weights.resolve()

    if not weights.exists():
        raise FileNotFoundError(weights)

    if not DATA_CONFIG.exists():
        raise FileNotFoundError(DATA_CONFIG)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GTSDB BASELINE EVALUATION")
    print("=" * 60)
    print(f"Weights: {weights}")
    print(f"Split: {args.split}")
    print(f"Device: {device}")

    model = YOLO(str(weights))

    metrics = model.val(
        data=str(DATA_CONFIG),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        plots=True,
        save_json=True,
        project=str(RESULTS_DIR),
        name="validation",
        exist_ok=True,
        verbose=True,
    )

    summary = {
        "model": "YOLOv8s",
        "weights": str(weights),
        "split": args.split,
        "image_size": args.imgsz,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": {},
    }

    for class_id, class_name in model.names.items():
        class_id = int(class_id)

        if class_id in metrics.box.ap_class_index:
            index = list(metrics.box.ap_class_index).index(class_id)

            summary["per_class"][str(class_id)] = {
                "name": str(class_name),
                "ap50_95": float(metrics.box.ap[index].mean()),
                "ap50": float(metrics.box.ap50[index]),
            }

    output_json = RESULTS_DIR / "metrics.json"
    output_json.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL METRICS")
    print("-" * 40)
    print(f"Precision:    {summary['precision']:.4f}")
    print(f"Recall:       {summary['recall']:.4f}")
    print(f"mAP@0.5:      {summary['map50']:.4f}")
    print(f"mAP@0.5:0.95: {summary['map50_95']:.4f}")

    import yaml

    config = yaml.safe_load(DATA_CONFIG.read_text())
    dataset_dir = Path(config["path"])

    if not dataset_dir.is_absolute():
        dataset_dir = (DATA_CONFIG.parent / dataset_dir).resolve()

    save_predictions(
        model=model,
        dataset_dir=dataset_dir,
        split=args.split,
        device=device,
        count=args.samples,
    )

    print("\nEvaluation completed.")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
