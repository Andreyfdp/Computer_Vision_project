import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from frcnn_dataset import GTSDBCocoDataset, collate_fn, resolve_images_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def build_model():
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, min_size=800, max_size=1360)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 44)
    return model


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    images_dir = resolve_images_dir(project_root, args.split, args.images_dir)
    ann = project_root / f"data/processed/instances_{args.split}.json"
    output = args.output if args.output.is_absolute() else project_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = GTSDBCocoDataset(images_dir, ann, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=collate_fn, pin_memory=device.type == "cuda")

    model = build_model()
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(device).eval()

    predictions = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    with torch.inference_mode():
        for images, targets in loader:
            images = [img.to(device, non_blocking=True) for img in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                image_id = int(target["image_id"].item())
                boxes = output["boxes"].detach().cpu()
                scores = output["scores"].detach().cpu()
                labels = output["labels"].detach().cpu()
                keep = scores >= 0.001
                for box, score, label in zip(boxes[keep], scores[keep], labels[keep]):
                    x1, y1, x2, y2 = box.tolist()
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(label.item()) - 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score.item()),
                    })

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    with output.open("w", encoding="utf-8") as f:
        json.dump(predictions, f)

    print(f"Saved predictions: {output}")
    print(f"Images: {len(ds)}")
    print(f"Predictions: {len(predictions)}")
    print(f"Wall time: {elapsed:.3f} s")
    print(f"ms/image: {elapsed / len(ds) * 1000:.3f}")
    print(f"images/s: {len(ds) / elapsed:.3f}")


if __name__ == "__main__":
    main()
