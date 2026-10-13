
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "FullIJCNN2013"
OUT = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"

NUM_CLASSES = 43
SEED = 42
WIDTH, HEIGHT = 1360, 800


def load_annotations():
    annotations = defaultdict(list)

    with (RAW / "gt.txt").open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            parts = line.strip().split(";")
            if len(parts) != 6:
                raise ValueError(f"Invalid annotation at line {line_number}")

            filename = parts[0]
            x1, y1, x2, y2, cls = map(int, parts[1:])

            if not (0 <= cls < NUM_CLASSES):
                raise ValueError(f"Invalid class {cls}")

            if not (0 <= x1 < x2 < WIDTH and 0 <= y1 < y2 < HEIGHT):
                raise ValueError(f"Invalid box at line {line_number}")

            if not (RAW / filename).is_file():
                raise FileNotFoundError(filename)

            annotations[filename].append((x1, y1, x2, y2, cls))

    return annotations


def create_splits(images):
    train_candidates = images[:600]
    test = images[600:]

    # Block-wise splitting: 20 blocks, 30 images each.
    blocks = [
        train_candidates[i:i + 30]
        for i in range(0, 600, 30)
    ]

    rng = random.Random(SEED)
    rng.shuffle(blocks)

    val = sorted(name for block in blocks[:4] for name in block)
    train = sorted(name for block in blocks[4:] for name in block)

    assert len(train) == 480
    assert len(val) == 120
    assert len(test) == 300

    return {"train": train, "val": val, "test": test}


def to_yolo(box):
    x1, y1, x2, y2, cls = box

    xc = ((x1 + x2) / 2) / WIDTH
    yc = ((y1 + y2) / 2) / HEIGHT
    w = (x2 - x1) / WIDTH
    h = (y2 - y1) / HEIGHT

    return f"{cls} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}"


def process_split(split_name, images, annotations):
    images_dir = OUT / "images" / split_name
    labels_dir = OUT / "labels" / split_name

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": str(i)}
            for i in range(NUM_CLASSES)
        ],
    }

    annotation_id = 1
    class_counts = Counter()

    for image_id, filename in enumerate(images, 1):
        source = RAW / filename
        target = images_dir / f"{Path(filename).stem}.png"

        if not target.exists():
            with Image.open(source) as image:
                if image.size != (WIDTH, HEIGHT):
                    raise ValueError(f"Unexpected image size: {filename}")
                image.convert("RGB").save(target)

        boxes = annotations.get(filename, [])

        label_file = labels_dir / f"{Path(filename).stem}.txt"
        label_file.write_text(
            "\n".join(to_yolo(box) for box in boxes),
            encoding="utf-8",
        )

        coco["images"].append({
            "id": image_id,
            "file_name": f"{Path(filename).stem}.png",
            "width": WIDTH,
            "height": HEIGHT,
        })

        for x1, y1, x2, y2, cls in boxes:
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cls,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })

            class_counts[cls] += 1
            annotation_id += 1

    (OUT / f"instances_{split_name}.json").write_text(
        json.dumps(coco, indent=2),
        encoding="utf-8",
    )

    return {
        "images": len(images),
        "objects": len(coco["annotations"]),
        "class_counts": dict(sorted(class_counts.items())),
    }


def main():
    images = sorted(p.name for p in RAW.glob("*.ppm"))

    if len(images) != 900:
        raise RuntimeError(f"Expected 900 images, found {len(images)}")

    annotations = load_annotations()
    splits = create_splits(images)

    OUT.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    statistics = {}

    for split_name, filenames in splits.items():
        (SPLITS_DIR / f"{split_name}.txt").write_text(
            "\n".join(filenames) + "\n",
            encoding="utf-8",
        )

        statistics[split_name] = process_split(
            split_name, filenames, annotations
        )

        print(
            f"{split_name}: "
            f"{statistics[split_name]['images']} images, "
            f"{statistics[split_name]['objects']} objects"
        )

    config = [
        f"path: '{OUT.as_posix()}'",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]

    for i in range(NUM_CLASSES):
        config.append(f"  {i}: '{i}'")

    (ROOT / "configs" / "gtsdb.yaml").write_text(
        "\n".join(config) + "\n",
        encoding="utf-8",
    )

    (OUT / "statistics.json").write_text(
        json.dumps(statistics, indent=2),
        encoding="utf-8",
    )

    print("\nPreprocessing complete.")
    print("YOLO config: configs/gtsdb.yaml")
    print("COCO annotations: data/processed/instances_*.json")


if __name__ == "__main__":
    main()
