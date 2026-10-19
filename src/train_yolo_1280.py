"""E2: YOLOv8s training at 1280px on GTSDB.

Run from the repository root:
    python src/train_yolo_1280.py --phase all --device 0 --batch 8

This is an independent experiment and never writes to the E0 directories
(runs/phase_a and runs/phase_b).
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = ROOT / "configs" / "gtsdb.yaml"
RUNS_DIR = ROOT / "runs"

PHASE_A_DIR = RUNS_DIR / "e2_phase_a"
PHASE_B_DIR = RUNS_DIR / "e2_phase_b"

SEED = 42
IMAGE_SIZE = 1280


def parse_args():
    parser = argparse.ArgumentParser(
        description="E2: YOLOv8s fine-tuning on GTSDB at 1280px"
    )
    parser.add_argument(
        "--phase", choices=["a", "b", "all"], default="all"
    )
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, 0, etc."
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs-a", type=int, default=5)
    parser.add_argument("--epochs-b", type=int, default=95)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--resume", choices=["a", "b"], default=None,
        help="Resume an interrupted E2 phase from its last.pt",
    )
    return parser.parse_args()


def resolve_device(requested):
    device = "0" if requested == "auto" and torch.cuda.is_available() else requested
    if device == "auto":
        device = "cpu"
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check PyTorch and NVIDIA drivers.")
    return device


def ensure_fresh_run(run_dir):
    """Avoid unintentionally overwriting an earlier E2 run."""
    if (run_dir / "results.csv").exists() or (run_dir / "weights" / "last.pt").exists():
        raise FileExistsError(
            f"An E2 run already exists in {run_dir}. "
            "Use --resume a/b for an interrupted run, or move the old "
            "directory before starting a new experiment."
        )


def common_train_args(args, device):
    return dict(
        data=str(DATA_CONFIG),
        imgsz=IMAGE_SIZE,
        batch=args.batch,
        device=device,
        workers=args.workers,
        seed=SEED,
        deterministic=True,
        optimizer="AdamW",
        cos_lr=True,
        fliplr=0.0,
        save=True,
        plots=True,
        project=str(RUNS_DIR),
        exist_ok=True,
    )


def train_phase_a(args, device):
    print("\n" + "=" * 60)
    print("E2 PHASE A: FROZEN BACKBONE (1280px)")
    print("=" * 60)

    ensure_fresh_run(PHASE_A_DIR)
    model = YOLO("yolov8s.pt")  # COCO pretraining, not the E0 checkpoint
    model.train(
        **common_train_args(args, device),
        name="e2_phase_a",
        epochs=args.epochs_a,
        freeze=10,
        lr0=0.001,
        save_period=1,
    )

    best = PHASE_A_DIR / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"E2 Phase A checkpoint missing: {best}")
    print(f"\nE2 Phase A best checkpoint: {best}")
    return best


def train_phase_b(args, device):
    print("\n" + "=" * 60)
    print("E2 PHASE B: FULL FINE-TUNING (1280px)")
    print("=" * 60)

    phase_a_best = PHASE_A_DIR / "weights" / "best.pt"
    if not phase_a_best.is_file():
        raise FileNotFoundError(
            f"Cannot start E2 Phase B: {phase_a_best} does not exist. "
            "Run --phase a first."
        )

    ensure_fresh_run(PHASE_B_DIR)
    model = YOLO(str(phase_a_best))
    model.train(
        **common_train_args(args, device),
        name="e2_phase_b",
        epochs=args.epochs_b,
        freeze=0,
        lr0=0.0001,
        warmup_epochs=3,
        patience=args.patience,
        save_period=5,
    )

    best = PHASE_B_DIR / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"E2 Phase B checkpoint missing: {best}")
    print(f"\nE2 Phase B best checkpoint: {best}")
    return best


def resume_training(args, device):
    phase_dir = PHASE_A_DIR if args.resume == "a" else PHASE_B_DIR
    checkpoint = phase_dir / "weights" / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"E2 resume checkpoint missing: {checkpoint}")

    print(f"Resuming E2 Phase {args.resume.upper()} from {checkpoint}")
    model = YOLO(str(checkpoint))
    model.train(resume=True, device=device)


def main():
    args = parse_args()
    if not DATA_CONFIG.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_CONFIG}")
    if args.batch < 1:
        raise ValueError("--batch must be a positive integer")

    device = resolve_device(args.device)
    print("=" * 60)
    print("GTSDB E2: YOLOv8s 1280px")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print(f"Device: {device}")
    if device != "cpu":
        print(f"GPU: {torch.cuda.get_device_name(int(device.split(',')[0]))}")
    print(f"Dataset: {DATA_CONFIG}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {args.batch}")
    print(f"Seed: {SEED}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if args.resume:
        resume_training(args, device)
        return
    if args.phase in ("a", "all"):
        train_phase_a(args, device)
    if args.phase in ("b", "all"):
        train_phase_b(args, device)

    print("\nE2 training completed successfully.")
    print(f"Final E2 weights: {PHASE_B_DIR / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
