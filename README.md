# Traffic Sign Detection on GTSDB

**Computer Vision 2026 — YOLOv8s baseline and higher-resolution experiment**

This project studies traffic-sign object detection on the **German Traffic Sign Detection Benchmark (GTSDB)**. We fine-tune a COCO-pretrained **YOLOv8s** detector to predict bounding boxes and labels for **43 traffic-sign classes**, then test whether increasing the input resolution improves the detection of small signs.

The repository currently contains two completed experimental conditions:

| ID | Model | Input size | Purpose | Status |
| --- | --- | ---: | --- | --- |
| **E0** | YOLOv8s, COCO-pretrained | 640 px | First working baseline | Completed |
| **E2** | YOLOv8s, COCO-pretrained | 1280 px | Resolution ablation / second approach | Completed |
| E1 | Faster R-CNN ResNet-50-FPN | Native resolution (planned) | Two-stage detector comparison | Not implemented yet |

E2 is a **significant modification of E0**, not a different network architecture. Its primary hypothesis is that retaining more image detail will improve small-object average precision (AP).

## Dataset

GTSDB contains **900 road-scene images** at **1360 × 800 pixels**. We use the original road scenes and their bounding-box annotations, not the separate cropped classification images.

| Split | Images | Annotated signs | Use |
| --- | ---: | ---: | --- |
| Train | 480 | 712 | Model fitting |
| Validation | 120 | 140 | Model selection and comparison |
| Test | 300 | 361 | Reserved for the final evaluation |

The split is deterministic (seed `42`). The first 600 images are assigned to train/validation using shuffled **contiguous blocks of 30 images** to reduce leakage from adjacent frames; the last 300 images remain the test set. All 43 classes appear in training, but only **28** appear in validation. Therefore, validation mAP does **not** establish performance on the 15 classes absent from the validation split.

`data/prepare.py` converts images from PPM to PNG, writes YOLO-format label files and COCO-format annotation JSON, and saves the split files in `data/splits/`.

## Environment and setup

The reported runs used:

| Component | Version / hardware |
| --- | --- |
| Python | 3.14.6 |
| PyTorch | 2.11.0+cu128 |
| Ultralytics | 8.4.177 |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB |
| OS | Windows |

From the repository root, install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

For GPU execution, install a CUDA-enabled PyTorch build compatible with your GPU and verify it:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`requirements.txt` lists the dependencies but does not pin versions; the versions above document the environment used for the reported results.

### Download and prepare GTSDB

Run **both commands from the repository root**:

```powershell
python data/download.py
python data/prepare.py
```

**Important:** `data/prepare.py` regenerates `configs/gtsdb.yaml` with an **absolute path** to `data/processed` on the current machine. The committed YAML contains a path from the training machine. After cloning on another computer, run `prepare.py` before training or evaluating so the path is corrected.

The raw dataset and processed images are not committed to Git.

## Training

Both models start independently from `yolov8s.pt` (COCO-pretrained weights), use the same dataset split and `seed=42`, and follow the same two-phase procedure:

- **Phase A:** freeze the first 10 YOLOv8 modules and train for 5 epochs with AdamW (`lr0=0.001`).
- **Phase B:** initialize from the best Phase A checkpoint, unfreeze the model, and fine-tune with AdamW (`lr0=0.0001`), cosine learning-rate scheduling and early stopping (`patience=20`), up to 95 epochs.

Horizontal flipping is disabled because it can alter the semantic meaning of directional signs. E2 uses a smaller batch because 1280-pixel images require more GPU memory.

| Setting | E0 (baseline) | E2 (second approach) |
| --- | ---: | ---: |
| Image size | 640 | 1280 |
| Batch size | 16 | 8 |
| Phase A epochs | 5 | 5 |
| Phase B completed epochs | 41 | 39 |
| Best Phase B epoch | 21 | 19 |
| Phase B training time reported by Ultralytics | 0.050 h | 0.163 h |
| Random seed | 42 | 42 |

The reported phase durations are **trainer-reported**, not total end-to-end times (which also include dataset preparation and other overhead).

### Train E0 (640 px)

```powershell
python -u src/train_yolo.py --phase all --device 0 --batch 16
```

Best checkpoint: `runs/phase_b/weights/best.pt`.

### Train E2 (1280 px)

```powershell
python -u src/train_yolo_1280.py --phase all --device 0 --batch 8
```

Best checkpoint: `runs/e2_phase_b/weights/best.pt`.

E2 uses separate run directories (`runs/e2_phase_a/`, `runs/e2_phase_b/`), so it does not overwrite E0. If the E2 directories already contain training results, the script prevents an accidental fresh rerun; use the documented `--resume a` / `--resume b` option only when resuming an interrupted run with a usable `last.pt`.

## Checkpoints

Model weight files (`*.pt`) and the `runs/` directory are excluded from Git.

- **E0:** [Download the baseline checkpoint (`best.pt`)](https://github.com/Andreyfdp/Computer_Vision_project/releases/download/baseline-v1.0/best.pt). Place it at `runs/phase_b/weights/best.pt`.
- **E2:** The trained checkpoint is `runs/e2_phase_b/weights/best.pt` on the training computer. **Publish this file as an asset of a GitHub Release tagged `e2-1280-v1.0`** if you want others to reproduce E2 without retraining. Check the [repository Releases page](https://github.com/Andreyfdp/Computer_Vision_project/releases) for availability.

Place each checkpoint at its expected path before running the corresponding evaluation command. The two files are independent models and must not be swapped.

## Reproduce validation evaluation

The same `src/evaluate.py` evaluates both models on the **same 120 validation images and 140 annotated signs**. It produces:

- `metrics.json` with precision, recall, Ultralytics mAP and per-class statistics;
- `validation/` with evaluation plots and COCO prediction JSON;
- `predictions/` with 20 annotated example images (prediction confidence threshold `0.25`).

Evaluation uses confidence threshold `0.001`, NMS IoU threshold `0.7` and `max_det=300`. The COCO-style summary printed by the evaluator uses `maxDets=100`. COCO AP and Ultralytics mAP are presented separately rather than treated as identical measurements.

### E0

```powershell
python -u src/evaluate.py --weights runs/phase_b/weights/best.pt --split val --imgsz 640 --batch 16 --device 0 --output-dir results/compare_e0
```

### E2

```powershell
python -u src/evaluate.py --weights runs/e2_phase_b/weights/best.pt --split val --imgsz 1280 --batch 8 --device 0 --output-dir results/compare_e2
```

On a CPU-only machine, substitute `--device cpu --batch 1` (the required checkpoints and processed dataset must still be available). Inference will be much slower; CPU timing must not be directly compared to the GPU measurements below.

## Results: E0 vs. E2

The following results are from the **validation set**, not from the reserved test set.

| Validation metric | E0: 640 px | E2: 1280 px | Difference (E2 − E0) |
| --- | ---: | ---: | ---: |
| **COCO AP@0.50:0.95** | **0.340** | **0.502** | **+0.162** |
| COCO AP@0.50 | 0.415 | 0.574 | +0.159 |
| **COCO AP small** | **0.275** | **0.505** | **+0.230** |
| COCO AP medium | 0.510 | 0.687 | +0.177 |
| COCO AP large | 0.767 | 0.680 | −0.087 |
| Precision (Ultralytics) | 0.4176 | 0.6365 | +0.2189 |
| Recall (Ultralytics) | 0.4035 | 0.5105 | +0.1070 |
| Ultralytics mAP@0.50:0.95 | 0.3377 | 0.4992 | +0.1615 |

**Finding:** Higher input resolution improves overall COCO AP by **0.162 absolute** (**47.6% relative**) and small-object AP by **0.230 absolute** (**83.6% relative**) on the validation split. This supports the small-sign resolution hypothesis, but **large-object AP decreases by 0.087**. Higher resolution does not improve every subset.

Ultralytics reported approximately **2.66 ms/image** of inference for E0 and **8.88 ms/image** for E2 on the same GPU. These figures came from validation runs with **different batch sizes (16 vs. 8)**; they indicate a speed cost but are **not** a controlled batch-1 latency benchmark.

### Saved evaluation artifacts

- [E0 metrics](results/compare_e0/metrics.json) · [E0 COCO predictions](results/compare_e0/validation/predictions.json)
- [E2 metrics](results/compare_e2/metrics.json) · [E2 COCO predictions](results/compare_e2/validation/predictions.json)

## Limitations

- The comparison uses **one training seed** for each condition; there is no estimate of run-to-run variance.
- E0 and E2 differ in **batch size** as well as input resolution, so the change is not perfectly isolated.
- The validation split is small and class-imbalanced, with **15 of 43 classes absent**. Per-class results for rare categories are unstable.
- AP for large signs decreased in E2; this remains a limitation rather than a claimed improvement.
- The reserved 300-image test set has **not** been evaluated. It is saved for the final project comparison.
- Faster R-CNN (E1) is planned but **not yet implemented** in this repository.
- Ultralytics reported and ignored one duplicate label for `00340.png` during training.

## Repository structure

```text
configs/gtsdb.yaml                 Dataset configuration (regenerated during preparation)
data/download.py                   GTSDB download script
data/prepare.py                    Dataset conversion, annotations and split generation
data/splits/                      Saved train/validation/test image lists
src/train_yolo.py                  E0: 640 px training
src/train_yolo_1280.py             E2: 1280 px training
src/evaluate.py                    Shared E0/E2 evaluation
results/compare_e0/               E0 validation metrics, plots, predictions
results/compare_e2/               E2 validation metrics, plots, predictions
requirements.txt                  Dependencies
```

## Team

- Vladislav Osetrov
- Andrei Ilin
- Matvej Shulaev
- Tamerlan Bagautdinov

**Course:** Computer Vision 2026. **Project:** Traffic Sign Detection on GTSDB.
