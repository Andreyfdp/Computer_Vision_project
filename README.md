# Traffic Sign Detection on GTSDB

Final Computer Vision course project on multi-class traffic-sign detection using the German Traffic Sign Detection Benchmark (GTSDB).

The project compares:

- **E0 — YOLOv8s, 640 px**
- **E1 — Faster R-CNN ResNet-50-FPN**
- **E2 — YOLOv8s, 1280 px**

The main research question is whether increasing input resolution improves detection of small traffic signs and how a one-stage YOLO detector compares with a two-stage Faster R-CNN detector.

---

## Dataset

We use the **German Traffic Sign Detection Benchmark (GTSDB)**.

- 900 road-scene images
- Original image size: `1360 × 800`
- 43 traffic-sign classes
- Bounding-box annotations
- Small and highly imbalanced objects

Final split:

| Split | Images | Objects |
|---|---:|---:|
| Train | 480 | 711 |
| Validation | 120 | 140 |
| Test | 300 | 361 |
| **Total** | **900** | **1212** |

One exact duplicate annotation was removed during preprocessing.

The train/validation split is block-wise in order to reduce leakage between visually similar neighboring frames.

The official final 300 images are used only for final evaluation.

---

## Experimental Setup

### E0 — YOLOv8s, 640 px

Baseline one-stage detector.

- YOLOv8s
- COCO pretrained
- Input resolution: `640`
- Two-stage fine-tuning
- AdamW
- Cosine learning-rate schedule
- No horizontal flip

### E1 — Faster R-CNN ResNet-50-FPN

Two-stage detector.

- Faster R-CNN
- ResNet-50-FPN backbone
- COCO pretrained
- Native GTSDB image scale
- Phase A: frozen backbone, 5 epochs
- Phase B: full fine-tuning, 30 epochs
- SGD
- Momentum: `0.9`
- Weight decay: `5e-4`
- No horizontal flip

### E2 — YOLOv8s, 1280 px

Resolution ablation.

- Same YOLOv8s architecture as E0
- Same COCO initialization
- Input resolution: `1280`
- Same dataset and evaluation protocol

This experiment isolates the effect of input resolution.

---

# Model Checkpoints and Releases

The best checkpoints for all three final experimental conditions are published through GitHub Releases.

| Experiment | Model | Release | Checkpoint |
|---|---|---|---|
| E0 | YOLOv8s, 640 px | `baseline-v1.0` | `best.pt` |
| E1 | Faster R-CNN ResNet-50-FPN | `e1-faster-rcnn-v1.0` | `best.pt` |
| E2 | YOLOv8s, 1280 px | `e2-1280-v1.0` | `best.pt` |

Each release contains the best checkpoint selected using validation performance.

### E0 — YOLOv8s 640

Release:

```text
baseline-v1.0
```

Checkpoint:

```text
best.pt
```

### E1 — Faster R-CNN ResNet-50-FPN

Release:

```text
e1-faster-rcnn-v1.0
```

Checkpoint:

```text
best.pt
```

The local checkpoint used to create this release is:

```text
runs/e1_seed0_phase_b/best.pt
```

### E2 — YOLOv8s 1280

Release:

```text
e2-1280-v1.0
```

Checkpoint:

```text
best.pt
```

The final test results reported below were obtained using these released checkpoints.

---

## Common Evaluation

All models are evaluated using the same COCO-style evaluator based on `pycocotools`.

Primary metric:

```text
AP@[IoU=0.50:0.95]
```

Additional metrics:

- AP50
- AP75
- AP for small objects
- AP for medium objects
- AP for large objects
- Average Recall
- per-class AP

Predictions from all models are converted to the same COCO JSON format before evaluation.

---

# Final Test Results

Final evaluation was performed on the official 300-image test set.

| Model | AP | AP50 | AP75 | AP Small | AP Medium | AP Large |
|---|---:|---:|---:|---:|---:|---:|
| E0 YOLOv8s 640 | 0.307 | 0.363 | 0.353 | 0.256 | 0.365 | 0.761 |
| E1 Faster R-CNN | 0.366 | 0.510 | 0.425 | 0.415 | 0.401 | 0.821 |
| **E2 YOLOv8s 1280** | **0.518** | **0.572** | **0.568** | **0.534** | **0.569** | **0.989** |

The strongest model is **E2 YOLOv8s at 1280 px**.

Compared with the 640-pixel baseline:

```text
Overall AP:
0.307 -> 0.518
relative improvement: +68.5%

Small-object AP:
0.256 -> 0.534
relative improvement: +108.8%
```

The results strongly support the hypothesis that input resolution is a major bottleneck for small traffic-sign detection on GTSDB.

---

# Runtime Benchmark

Runtime measurements were performed on:

```text
NVIDIA GeForce RTX 5060 Ti 16 GB
batch size = 1
5 warm-up runs
50 measured runs
```

| Model | Latency, ms/image | FPS | Parameters | Peak VRAM |
|---|---:|---:|---:|---:|
| E0 YOLOv8s 640 | **22.42** | **44.60** | 11.15 M | **68 MB** |
| E1 Faster R-CNN | 46.35 | 21.57 | 41.57 M | 740 MB |
| E2 YOLOv8s 1280 | 25.10 | 39.85 | 11.15 M | 163 MB |

E2 increases latency by only approximately **11.9%** relative to E0 while increasing AP by **68.5%**.

Faster R-CNN is approximately **1.85× slower than E2**, has almost four times more parameters, and uses substantially more GPU memory.

Therefore, E2 provides the best accuracy-efficiency trade-off.

**Timing methodology note:** YOLO timing includes image loading and preprocessing, whereas Faster R-CNN timing excludes them. Therefore, cross-model latency and throughput comparisons should be treated as approximate rather than strictly equivalent end-to-end measurements.

---

# Failure Analysis

The final test predictions were analyzed using several error categories:

- correct detection
- classification error
- localization error
- missed ground truth
- background false positive

Three representative failure modes were identified.

### 1. Classification error

The detector can localize a traffic sign almost perfectly but assign the wrong class.

This mainly occurs between visually similar traffic-sign classes that differ only by a small symbol or number.

### 2. Missed small object

Very small traffic signs can still be missed even at 1280-pixel input resolution.

This occurs especially when:

- the sign occupies very few pixels;
- the background is visually cluttered;
- contrast is low.

### 3. Localization error

The detector identifies the correct class but predicts an inaccurate bounding box.

This was observed in Faster R-CNN examples with nearby signs or surrounding structures.

Overall, the main remaining limitations are:

- extremely small objects;
- visually similar classes;
- occasional localization errors.

---

# Main Conclusion

The best model is:

```text
YOLOv8s
input resolution: 1280
AP@[0.50:0.95]: 0.518
AP50: 0.572
AP Small: 0.534
FPS: 39.85
```

The controlled comparison between E0 and E2 demonstrates that increasing the input resolution from 640 to 1280 produces a large improvement in detection quality without increasing model size.

For GTSDB, preserving spatial detail is more important than replacing YOLOv8s with a substantially heavier two-stage detector.

---

# Repository Structure

```text
Computer_Vision_project/
│
├── configs/
│   └── gtsdb.yaml
│
├── data/
│   ├── download.py
│   ├── prepare.py
│   └── processed/
│
├── src/
│   ├── train_yolo.py
│   ├── train_yolo_1280.py
│   ├── evaluate.py
│   ├── audit_coco_results.py
│   ├── frcnn_dataset.py
│   ├── train_frcnn.py
│   ├── predict_frcnn.py
│   ├── benchmark_runtime.py
│   └── failure_analysis.py
│
├── runs/
│   ├── phase_b/
│   ├── e2_phase_b/
│   ├── e1_seed0_phase_a/
│   └── e1_seed0_phase_b/
│
├── results/
│   ├── test_e0/
│   ├── test_e1/
│   ├── test_e2/
│   ├── failure_analysis/
│   └── runtime_benchmark.json
│
└── README.md
```

---

# Reproduction

## 1. Environment

Install project dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
Python
PyTorch
torchvision
Ultralytics
pycocotools
OpenCV
Pillow
NumPy
pandas
```

---

## 2. Prepare Dataset

Download the GTSDB dataset:

```bash
python data/download.py
```

Then run preprocessing:

```bash
python data/prepare.py
```

Expected result:

```text
train: 480 images, 711 objects
val: 120 images, 140 objects
test: 300 images, 361 objects

Exact duplicate annotations removed: 1
```

---

## 3. Evaluate E0

```bash
python src/evaluate.py \
  --weights runs/phase_b/weights/best.pt \
  --split test \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --samples 0 \
  --output-dir results/test_e0
```

```bash
python src/audit_coco_results.py \
  --split test \
  --predictions results/test_e0/coco_predictions.json \
  --output results/test_e0/coco_audit.json
```

---

## 4. Evaluate E2

```bash
python src/evaluate.py \
  --weights runs/e2_phase_b/weights/best.pt \
  --split test \
  --imgsz 1280 \
  --batch 1 \
  --device 0 \
  --samples 0 \
  --output-dir results/test_e2
```

```bash
python src/audit_coco_results.py \
  --split test \
  --predictions results/test_e2/coco_predictions.json \
  --output results/test_e2/coco_audit.json
```

---

## 5. Evaluate Faster R-CNN

```bash
python src/predict_frcnn.py \
  --weights runs/e1_seed0_phase_b/best.pt \
  --split test \
  --device cuda:0 \
  --batch-size 1 \
  --output results/test_e1/coco_predictions.json
```

```bash
python src/audit_coco_results.py \
  --split test \
  --predictions results/test_e1/coco_predictions.json \
  --output results/test_e1/coco_audit.json
```

---

## 6. Runtime Benchmark

```bash
python src/benchmark_runtime.py \
  --device cuda:0 \
  --warmup 5 \
  --runs 50 \
  --output results/runtime_benchmark.json
```

---

## 7. Failure Analysis

```bash
python src/failure_analysis.py \
  --split test \
  --conf 0.25 \
  --output-dir results/failure_analysis
```

---

# Team

| Student | Main responsibility |
|---|---|
| Vladislav Osetrov | Dataset preparation, annotation conversion, splits |
| Andrei Ilin | YOLOv8 E0/E2 experiments and resolution ablation |
| Matvej Shulaev | Faster R-CNN E1 and efficiency evaluation |
| Tamerlan Bagautdinov | Common evaluation, metrics, failure analysis |

---

# Final Result

**Best model: YOLOv8s at 1280 px**

```text
COCO AP:      0.518
AP50:         0.572
AP75:         0.568
AP Small:     0.534
AP Medium:    0.569
AP Large:     0.989
Latency:      25.10 ms/image
Throughput:   39.85 FPS
Parameters:   11.15 M
```