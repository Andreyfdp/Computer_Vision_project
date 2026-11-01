import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from frcnn_dataset import GTSDBCocoDataset, collate_fn, resolve_images_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["a", "b"], required=True)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--train-json", type=Path, default=Path("data/processed/instances_train.json"))
    p.add_argument("--val-json", type=Path, default=Path("data/processed/instances_val.json"))
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes=44):
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        min_size=800,
        max_size=1360,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def set_phase_trainability(model, phase):
    for p in model.parameters():
        p.requires_grad = True
    if phase == "a":
        for p in model.backbone.parameters():
            p.requires_grad = False


def export_predictions(model, loader, device):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for images, targets in loader:
            images = [img.to(device) for img in images]
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
    return predictions


def evaluate_coco(coco_gt, predictions):
    if not predictions:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0}
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    s = evaluator.stats
    return {
        "AP": float(s[0]),
        "AP50": float(s[1]),
        "AP75": float(s[2]),
        "AP_small": float(s[3]),
        "AP_medium": float(s[4]),
        "AP_large": float(s[5]),
    }


def save_checkpoint(path, model, optimizer, epoch, best_ap, args):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_ap": best_ap,
        "args": vars(args),
    }, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    project_root = Path(__file__).resolve().parents[1]
    images_dir = args.images_dir or resolve_images_dir(project_root)
    train_json = args.train_json if args.train_json.is_absolute() else project_root / args.train_json
    val_json = args.val_json if args.val_json.is_absolute() else project_root / args.val_json
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs if args.epochs is not None else (5 if args.phase == "a" else 30)
    lr = args.lr if args.lr is not None else (0.005 if args.phase == "a" else 0.001)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    train_ds = GTSDBCocoDataset(images_dir, train_json, train=True, seed=args.seed)
    val_ds = GTSDBCocoDataset(images_dir, val_json, train=False, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn, pin_memory=device.type == "cuda")

    model = build_model()
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded checkpoint: {args.resume}")

    set_phase_trainability(model, args.phase)
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max(1, int(epochs * 0.67)), max(2, int(epochs * 0.9))],
        gamma=0.1,
    )

    history_path = output_dir / "history.csv"
    if not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "lr", "epoch_seconds"])

    best_ap = -1.0
    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            losses = model(images, targets)
            loss = sum(losses.values())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

            if step % 20 == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{epochs} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / max(1, len(train_loader))

        predictions = export_predictions(model, val_loader, device)
        metrics = evaluate_coco(val_ds.coco, predictions)
        epoch_seconds = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        with history_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch, train_loss, metrics["AP"], metrics["AP50"], metrics["AP75"],
                metrics["AP_small"], metrics["AP_medium"], metrics["AP_large"], current_lr, epoch_seconds,
            ])

        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_ap, args)
        if metrics["AP"] > best_ap:
            best_ap = metrics["AP"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_ap, args)
            with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
                json.dump({"epoch": epoch, **metrics}, f, indent=2)
            with (output_dir / "best_predictions.json").open("w", encoding="utf-8") as f:
                json.dump(predictions, f)

        print(
            f"Epoch {epoch}: loss={train_loss:.4f} AP={metrics['AP']:.4f} "
            f"AP50={metrics['AP50']:.4f} AP75={metrics['AP75']:.4f} "
            f"APS={metrics['AP_small']:.4f} best={best_ap:.4f}"
        )

    total_seconds = time.perf_counter() - total_start
    summary = {
        "phase": args.phase,
        "seed": args.seed,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "lr": lr,
        "device": str(device),
        "best_AP": best_ap,
        "training_wall_time_seconds": total_seconds,
        "training_wall_time_minutes": total_seconds / 60.0,
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
