import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--e0', type=Path, default=Path('runs/phase_b/weights/best.pt'))
    p.add_argument('--e1', type=Path, default=Path('runs/e1_seed0_phase_b/best.pt'))
    p.add_argument('--e2', type=Path, default=Path('runs/e2_phase_b/weights/best.pt'))
    p.add_argument('--images-dir', type=Path, default=Path('data/processed/images/test'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--runs', type=int, default=50)
    p.add_argument('--output', type=Path, default=Path('results/runtime_benchmark.json'))
    return p.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def summarize(name, latencies_ms, params, checkpoint_bytes, peak_vram_bytes, input_desc):
    return {
        'model': name,
        'batch_size': 1,
        'input': input_desc,
        'runs': len(latencies_ms),
        'latency_ms_mean': statistics.mean(latencies_ms),
        'latency_ms_median': statistics.median(latencies_ms),
        'latency_ms_p95': percentile(latencies_ms, 0.95),
        'throughput_images_per_second': 1000.0 / statistics.mean(latencies_ms),
        'parameters': int(params),
        'checkpoint_size_mb': checkpoint_bytes / (1024 ** 2),
        'peak_vram_mb': peak_vram_bytes / (1024 ** 2),
    }


def benchmark_yolo(weights_path, image_paths, imgsz, device):
    model = YOLO(str(weights_path))
    for path in image_paths[:ARGS.warmup]:
        model.predict(source=str(path), imgsz=imgsz, batch=1, device=ARGS.device,
                      conf=0.001, iou=0.7, max_det=300, verbose=False)
    sync(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    latencies = []
    for path in image_paths[:ARGS.runs]:
        sync(device)
        t0 = time.perf_counter()
        result = model.predict(source=str(path), imgsz=imgsz, batch=1, device=ARGS.device,
                               conf=0.001, iou=0.7, max_det=300, verbose=False)
        sync(device)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        del result

    params = sum(p.numel() for p in model.model.parameters())
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    summary = summarize(
        f'YOLOv8s-{imgsz}', latencies, params, weights_path.stat().st_size, peak, f'{imgsz}px letterbox'
    )
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return summary


def build_frcnn():
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=800,
        max_size=1360,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 44)
    return model


def load_image_tensor(path):
    with Image.open(path) as img:
        img = img.convert('RGB')
        return TF.pil_to_tensor(img).float() / 255.0


def benchmark_frcnn(weights_path, image_paths, device):
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    model = build_frcnn()
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()

    for path in image_paths[:ARGS.warmup]:
        image = load_image_tensor(path).to(device)
        with torch.inference_mode():
            output = model([image])
        del image, output
    sync(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    latencies = []
    for path in image_paths[:ARGS.runs]:
        image = load_image_tensor(path).to(device)
        sync(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = model([image])
        sync(device)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        del image, output

    params = sum(p.numel() for p in model.parameters())
    peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    summary = summarize(
        'Faster R-CNN R50-FPN', latencies, params, weights_path.stat().st_size, peak, 'native 800x1360 policy'
    )
    del model, checkpoint
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return summary


def main():
    global ARGS
    ARGS = parse_args()
    if ARGS.warmup < 1 or ARGS.runs < 1:
        raise ValueError('--warmup and --runs must be positive')

    device = torch.device(ARGS.device if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('Runtime benchmark must be run on the GPU machine.')

    e0 = resolve(ARGS.e0)
    e1 = resolve(ARGS.e1)
    e2 = resolve(ARGS.e2)
    images_dir = resolve(ARGS.images_dir)
    output = resolve(ARGS.output)

    for p in [e0, e1, e2]:
        if not p.is_file():
            raise FileNotFoundError(p)
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)

    image_paths = sorted(images_dir.glob('*.png'))
    required = max(ARGS.warmup, ARGS.runs)
    if len(image_paths) < required:
        raise RuntimeError(f'Need at least {required} test images, found {len(image_paths)}')

    torch.backends.cudnn.benchmark = True
    results = {
        'gpu': torch.cuda.get_device_name(device),
        'torch': torch.__version__,
        'warmup_images': ARGS.warmup,
        'measured_images': ARGS.runs,
        'protocol': 'batch=1; CUDA synchronize before/after timed inference; model loading excluded; warm-up excluded',
        'models': [],
    }

    results['models'].append(benchmark_yolo(e0, image_paths, 640, device))
    results['models'].append(benchmark_frcnn(e1, image_paths, device))
    results['models'].append(benchmark_yolo(e2, image_paths, 1280, device))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
