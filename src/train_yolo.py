
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]

DATA_CONFIG = ROOT / "configs" / "gtsdb.yaml"
RUNS_DIR = ROOT / "runs"

PHASE_A_DIR = RUNS_DIR / "phase_a"
PHASE_B_DIR = RUNS_DIR / "phase_b"

SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8s fine-tuning on GTSDB"
    )

    parser.add_argument(
        "--phase",
        choices=["a", "b", "all"],
        default="all",
        help="Training phase: frozen backbone, full fine-tuning, or both",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, cpu, 0, etc.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--epochs-a",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--epochs-b",
        type=int,
        default=95,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--resume",
        choices=["a", "b"],
        default=None,
        help="Resume an interrupted training phase from last.pt",
    )

    return parser.parse_args()


def resolve_device(device):
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"

    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check PyTorch and NVIDIA drivers."
        )

    return device


def train_phase_a(args, device):
    print("\n" + "=" * 60)
    print("PHASE A: FROZEN BACKBONE")
    print("=" * 60)

    model = YOLO("yolov8s.pt")

    model.train(
        data=str(DATA_CONFIG),
        epochs=args.epochs_a,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,

        seed=SEED,
        deterministic=True,

        freeze=10,

        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,

        fliplr=0.0,

        save=True,
        save_period=1,
        plots=True,

        project=str(RUNS_DIR),
        name="phase_a",
        exist_ok=True,
    )

    best_weights = PHASE_A_DIR / "weights" / "best.pt"

    if not best_weights.exists():
        raise FileNotFoundError(best_weights)

    print(f"\nPhase A completed: {best_weights}")

    return best_weights


def train_phase_b(args, device):
    print("\n" + "=" * 60)
    print("PHASE B: FULL FINE-TUNING")
    print("=" * 60)

    phase_a_weights = PHASE_A_DIR / "weights" / "best.pt"

    if not phase_a_weights.exists():
        raise FileNotFoundError(
            "Phase A checkpoint not found. "
            "Run Phase A before Phase B."
        )

    model = YOLO(str(phase_a_weights))

    model.train(
        data=str(DATA_CONFIG),
        epochs=args.epochs_b,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,

        seed=SEED,
        deterministic=True,

        freeze=0,

        optimizer="AdamW",
        lr0=0.0001,
        cos_lr=True,
        warmup_epochs=3,

        patience=args.patience,

        fliplr=0.0,

        save=True,
        save_period=5,
        plots=True,

        project=str(RUNS_DIR),
        name="phase_b",
        exist_ok=True,
    )

    best_weights = PHASE_B_DIR / "weights" / "best.pt"

    if not best_weights.exists():
        raise FileNotFoundError(best_weights)

    print(f"\nPhase B completed: {best_weights}")

    return best_weights


def resume_training(args, device):
    phase_dir = PHASE_A_DIR if args.resume == "a" else PHASE_B_DIR
    checkpoint = phase_dir / "weights" / "last.pt"

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint available for resume: {checkpoint}"
        )

    print(f"Resuming training from {checkpoint}")

    model = YOLO(str(checkpoint))
    model.train(resume=True, device=device)


def main():
    args = parse_args()

    if not DATA_CONFIG.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {DATA_CONFIG}"
        )

    device = resolve_device(args.device)

    print("=" * 60)
    print("GTSDB TRAFFIC SIGN DETECTION")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")

    if device != "cpu":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Dataset: {DATA_CONFIG}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Random seed: {SEED}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if args.resume:
        resume_training(args, device)
        return

    if args.phase in ("a", "all"):
        train_phase_a(args, device)

    if args.phase in ("b", "all"):
        train_phase_b(args, device)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
