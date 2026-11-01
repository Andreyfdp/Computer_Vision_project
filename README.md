# E1 Faster R-CNN ResNet-50-FPN

Copy the three files from `src/` into the project's `src/` directory.

Training recipe follows the original project plan:
- Faster R-CNN ResNet-50-FPN
- COCO pretrained
- 43 GTSDB classes + background
- native GTSDB scale through min_size=800, max_size=1360
- Phase A: frozen backbone, 5 epochs
- Phase B: full fine-tuning, 30 epochs
- SGD, momentum 0.9, weight decay 5e-4
- no horizontal flip
- model selection by validation COCO AP@[0.5:0.95]

Commands are provided in the chat response.
