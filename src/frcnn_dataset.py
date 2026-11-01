from pathlib import Path
import random

import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as TF


class GTSDBCocoDataset(CocoDetection):
    def __init__(self, images_dir, annotation_file, train=False, seed=42):
        super().__init__(root=str(images_dir), annFile=str(annotation_file))
        self.train = train
        self.rng = random.Random(seed)

    def __getitem__(self, index):
        image, annotations = super().__getitem__(index)
        image_id = self.ids[index]

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann["category_id"]) + 1)
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        if self.train:
            image = self._augment(image)

        image = TF.pil_to_tensor(image).float() / 255.0

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
        }
        return image, target

    def _augment(self, image):
        if self.rng.random() < 0.8:
            image = TF.adjust_brightness(image, self.rng.uniform(0.85, 1.15))
        if self.rng.random() < 0.8:
            image = TF.adjust_contrast(image, self.rng.uniform(0.85, 1.15))
        if self.rng.random() < 0.4:
            image = TF.adjust_saturation(image, self.rng.uniform(0.9, 1.1))
        if self.rng.random() < 0.2:
            image = TF.gaussian_blur(image, kernel_size=[3, 3], sigma=self.rng.uniform(0.1, 1.0))
        return image


def collate_fn(batch):
    return tuple(zip(*batch))


def resolve_images_dir(project_root: Path) -> Path:
    candidates = [
        project_root / "data" / "processed" / "images",
        project_root / "data" / "processed",
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.png")):
            return candidate
    raise FileNotFoundError(
        "Could not find processed PNG images. Expected data/processed/images/*.png "
        "or data/processed/*.png. Pass --images-dir explicitly."
    )
