"""Re-evaluate saved YOLO COCO predictions without loading or training a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-yaml", default="configs/gtsdb.yaml")
    p.add_argument("--split", choices=("train", "val", "test"), required=True)
    p.add_argument("--predictions", required=True, help="Saved coco_predictions.json")
    p.add_argument("--output", required=True, help="Output JSON path")
    return p.parse_args()


def resolve_dataset_dir(yaml_path: Path) -> Path:
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root


def summarize(coco_gt: COCO, predictions: list[dict]) -> dict:
    image_ids = {int(x["id"]) for x in coco_gt.dataset["images"]}
    category_ids = {int(x["id"]) for x in coco_gt.dataset["categories"]}
    file_names = {str(x["file_name"]) for x in coco_gt.dataset["images"]}
    if not predictions:
        raise ValueError("Prediction file is empty")

    for i, p in enumerate(predictions):
        required = {"image_id", "category_id", "bbox", "score"}
        if not required.issubset(p):
            raise ValueError(f"Prediction {i} misses keys: {required - set(p)}")
        if int(p["image_id"]) not in image_ids:
            raise ValueError(f"Prediction {i} has unknown image_id {p['image_id']}")
        if int(p["category_id"]) not in category_ids:
            raise ValueError(f"Prediction {i} has unknown category_id {p['category_id']}")
        if len(p["bbox"]) != 4 or not np.isfinite(p["bbox"]).all():
            raise ValueError(f"Prediction {i} has invalid bbox")
        x, y, w, h = map(float, p["bbox"])
        if w <= 0 or h <= 0 or not np.isfinite(float(p["score"])):
            raise ValueError(f"Prediction {i} has invalid size or score")

    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.maxDets = [1, 10, 100]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    # Capture the actual IoU thresholds and the AP tensor dimensions. This makes
    # it possible to diagnose suspiciously identical AP50/AP75 values.
    iou_thresholds = [float(x) for x in ev.params.iouThrs]
    if len(iou_thresholds) != 10 or not np.allclose(iou_thresholds, np.linspace(.5, .95, 10)):
        raise RuntimeError(f"Unexpected COCO IoU thresholds: {iou_thresholds}")

    per_class = {}
    for category in coco_gt.dataset["categories"]:
        cid = int(category["id"])
        class_ev = COCOeval(coco_gt, coco_dt, "bbox")
        class_ev.params.maxDets = [1, 10, 100]
        class_ev.params.catIds = [cid]
        class_ev.evaluate()
        class_ev.accumulate()
        precision = class_ev.eval["precision"]  # [IoU, recall, category, area, maxDet]
        valid = precision[:, :, 0, 0, 2]
        valid = valid[valid > -1]
        ap = float(np.mean(valid)) if valid.size else None
        p50 = precision[0, :, 0, 0, 2]
        p50 = p50[p50 > -1]
        ap50 = float(np.mean(p50)) if p50.size else None
        per_class[str(cid)] = {"name": str(category.get("name", cid)), "AP": ap, "AP50": ap50}

    stats = ev.stats
    return {
        "images": len(coco_gt.dataset["images"]),
        "ground_truth_annotations": len(coco_gt.dataset["annotations"]),
        "predictions": len(predictions),
        "categories": len(category_ids),
        "iou_thresholds": iou_thresholds,
        "max_detections": [1, 10, 100],
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR_1": float(stats[6]),
        "AR_10": float(stats[7]),
        "AR_100": float(stats[8]),
        "per_class": per_class,
        "diagnostics": {
            "ap50_equals_ap75": bool(np.isclose(stats[1], stats[2], atol=1e-12)),
            "ap_tensor_shape": list(ev.eval["precision"].shape),
            "note": "AP50 and AP75 can legitimately be equal; inspect per-class results and predictions before treating equality as a bug.",
        },
    }


def main():
    args = parse_args()
    yaml_path = Path(args.data_yaml).resolve()
    pred_path = Path(args.predictions).resolve()
    output_path = Path(args.output).resolve()
    if not yaml_path.is_file() or not pred_path.is_file():
        raise FileNotFoundError(f"Missing input: yaml={yaml_path.is_file()}, predictions={pred_path.is_file()}")
    gt_path = resolve_dataset_dir(yaml_path) / f"instances_{args.split}.json"
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    coco_gt = COCO(str(gt_path))
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    result = summarize(coco_gt, predictions)
    result["split"] = args.split
    result["ground_truth_file"] = str(gt_path)
    result["prediction_file"] = str(pred_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_class"}, indent=2))
    print(f"Saved full audit (including per-class AP): {output_path}")


if __name__ == "__main__":
    main()
