"""Evaluate a trained YOLO checkpoint on GTSDB.

The script reports Ultralytics metrics and an independent COCOeval result.
COCO image IDs are resolved from the ground-truth JSON by file name, and
YOLO class indices are mapped explicitly to the category IDs in that JSON.
No training is performed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = ROOT / "configs" / "gtsdb.yaml"
DEFAULT_WEIGHTS = ROOT / "runs" / "phase_b" / "weights" / "best.pt"
DEFAULT_OUTPUT = ROOT / "results" / "evaluation"


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO on GTSDB")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, ...")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_device(value):
    if value == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if value != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu instead.")
    return value


def load_dataset_config():
    local_data_dir = (ROOT / "data" / "processed").resolve()
    local_annotation = local_data_dir / f"instances_{ARGS.split}.json"
    local_images = local_data_dir / "images" / ARGS.split

    if local_annotation.is_file() and local_images.is_dir():
        return local_data_dir, local_annotation, local_images

    if not DATA_CONFIG.is_file():
        raise FileNotFoundError(
            f"Dataset not found under {local_data_dir} and configuration not found: {DATA_CONFIG}"
        )

    with DATA_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    configured_data_dir = Path(config["path"])
    if not configured_data_dir.is_absolute():
        configured_data_dir = (DATA_CONFIG.parent / configured_data_dir).resolve()

    annotation_file = configured_data_dir / f"instances_{ARGS.split}.json"
    image_dir = configured_data_dir / "images" / ARGS.split
    if not annotation_file.is_file():
        raise FileNotFoundError(
            f"COCO annotations not found. Checked {local_annotation} and {annotation_file}"
        )
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found. Checked {local_images} and {image_dir}"
        )
    return configured_data_dir, annotation_file, image_dir


def build_id_maps(coco_gt):
    """Return filename->image_id and category IDs from the GT JSON."""
    images = coco_gt.dataset.get("images", [])
    categories = coco_gt.dataset.get("categories", [])
    filename_to_id = {Path(item["file_name"]).name: int(item["id"]) for item in images}
    category_ids = {int(item["id"]) for item in categories}
    if len(filename_to_id) != len(images):
        raise ValueError("Duplicate file_name values in COCO ground truth")
    if not category_ids:
        raise ValueError("No categories found in COCO ground truth")
    return filename_to_id, category_ids


def collect_predictions(model, image_paths, filename_to_id, category_ids,
                        device, imgsz, batch, conf, nms_iou, max_det,
                        output_dir, samples):
    predictions = []
    sample_dir = output_dir / "predictions"
    if samples > 0:
        sample_dir.mkdir(parents=True, exist_ok=True)

    selected_samples = set()
    if samples > 0 and image_paths:
        rng = np.random.default_rng(42)
        selected_samples = {
            image_paths[int(i)].name
            for i in rng.choice(len(image_paths), size=min(samples, len(image_paths)), replace=False)
        }

    start = time.perf_counter()
    seen = 0
    for chunk_start in range(0, len(image_paths), batch):
        chunk = image_paths[chunk_start:chunk_start + batch]
        results = model.predict(
            source=[str(path) for path in chunk],
            imgsz=imgsz,
            batch=len(chunk),
            device=device,
            conf=conf,
            iou=nms_iou,
            max_det=max_det,
            stream=False,
            verbose=False,
        )
        if len(results) != len(chunk):
            raise RuntimeError(
                f"Ultralytics returned {len(results)} results for a chunk of {len(chunk)} images"
            )
        for result in results:
            filename = Path(result.path).name
            if filename not in filename_to_id:
                raise KeyError(f"Prediction image is absent from COCO GT: {filename}")
            image_id = filename_to_id[filename]
            boxes = result.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.detach().cpu().numpy()
                scores = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                for box, score, class_id in zip(xyxy, scores, classes):
                    category_id = int(class_id)
                    if category_id not in category_ids:
                        raise ValueError(
                            f"Model class {category_id} is not a COCO category. "
                            "Check the model/data class mapping."
                        )
                    x1, y1, x2, y2 = map(float, box)
                    predictions.append({
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    })
            if filename in selected_samples:
                result.save(filename=str(sample_dir / filename))
            seen += 1
            if seen % 50 == 0 or seen == len(image_paths):
                print(f"Inference: {seen}/{len(image_paths)} images")

        del results
        if torch.cuda.is_available() and str(device) != "cpu":
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - start
    if seen != len(image_paths):
        raise RuntimeError(f"Processed {seen} images, expected {len(image_paths)}")
    return predictions, elapsed, seen


def evaluate_coco(annotation_file, predictions, output_dir):
    prediction_file = output_dir / "coco_predictions.json"
    prediction_file.write_text(json.dumps(predictions), encoding="utf-8")
    if not predictions:
        raise RuntimeError("No detections were produced; COCOeval cannot run")

    coco_gt = COCO(str(annotation_file))
    coco_dt = coco_gt.loadRes(str(prediction_file))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
    }


def main():
    global ARGS
    ARGS = parse_args()
    args = ARGS
    weights = project_path(args.weights)
    output_dir = project_path(args.output_dir)
    device = resolve_device(args.device)

    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {weights}")
    if args.imgsz <= 0 or args.batch <= 0 or args.max_det <= 0:
        raise ValueError("imgsz, batch and max-det must be positive")
    if not 0 <= args.conf < 1 or not 0 < args.nms_iou <= 1:
        raise ValueError("conf must be in [0,1), and nms-iou in (0,1]")
    if args.samples < 0:
        raise ValueError("samples must be nonnegative")

    output_dir.mkdir(parents=True, exist_ok=True)
    _, annotation_file, image_dir = load_dataset_config()
    coco_gt = COCO(str(annotation_file))
    filename_to_id, category_ids = build_id_maps(coco_gt)
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"No PNG images found in {image_dir}")
    missing = [p.name for p in image_paths if p.name not in filename_to_id]
    if missing:
        raise ValueError(f"Images missing from COCO GT (first 10): {missing[:10]}")
    if len(image_paths) != len(filename_to_id):
        raise ValueError(
            f"Image count mismatch: directory={len(image_paths)}, COCO={len(filename_to_id)}"
        )

    print(f"Checkpoint: {weights}")
    print(f"Split: {args.split}; images: {len(image_paths)}; annotations: {len(coco_gt.anns)}")
    print(f"Image size: {args.imgsz}; batch: {args.batch}; device: {device}")

    model = YOLO(str(weights))
    model_names = {int(k): str(v) for k, v in model.names.items()}
    if set(model_names) != category_ids:
        raise ValueError(
            "Model class IDs and COCO category IDs differ: "
            f"model={sorted(model_names)}, coco={sorted(category_ids)}. "
            "Do not evaluate until the mapping is explicitly resolved."
        )

    predictions, elapsed, seen = collect_predictions(
        model, image_paths, filename_to_id, category_ids,
        device, args.imgsz, args.batch, args.conf, args.nms_iou,
        args.max_det, output_dir, args.samples,
    )
    coco_metrics = evaluate_coco(annotation_file, predictions, output_dir)

    summary = {
        "model": "YOLO",
        "weights": str(weights),
        "split": args.split,
        "images": seen,
        "ground_truth_annotations": len(coco_gt.anns),
        "image_size": args.imgsz,
        "batch_size": args.batch,
        "device": str(device),
        "confidence_threshold": args.conf,
        "nms_iou": args.nms_iou,
        "max_detections": args.max_det,
        "prediction_count": len(predictions),
        "inference_wall_time_seconds": elapsed,
        "inference_wall_time_ms_per_image": elapsed * 1000 / seen,
        "throughput_images_per_second": seen / elapsed if elapsed else None,
        "coco": coco_metrics,
        "class_id_mapping": {str(k): k for k in sorted(model_names)},
        "notes": [
            "COCO image IDs are looked up by file_name from instances_<split>.json.",
            "Model class indices are matched explicitly to COCO category IDs.",
            "COCOeval uses bbox IoU thresholds 0.50:0.95 and maxDets [1,10,100].",
            "Inference wall time includes prediction iteration but excludes model loading and COCO evaluation.",
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
