from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def iou_xywh(a, b):
    ax1, ay1, aw, ah = map(float, a)
    bx1, by1, bw, bh = map(float, b)
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def resolve_project_root():
    return Path(__file__).resolve().parents[1]


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def classify_errors(gt_by_image, pred_by_image, iou_match=0.5, iou_loose=0.1):
    errors = []
    counts = Counter()

    for image_id, gts in gt_by_image.items():
        preds = sorted(pred_by_image.get(image_id, []), key=lambda x: float(x.get('score', 0.0)), reverse=True)
        used_gt = set()
        used_pred = set()

        # Correct matches first: same class + IoU >= threshold.
        candidates = []
        for pi, p in enumerate(preds):
            for gi, g in enumerate(gts):
                if int(p['category_id']) == int(g['category_id']):
                    iou = iou_xywh(p['bbox'], g['bbox'])
                    if iou >= iou_match:
                        candidates.append((iou, float(p.get('score', 0.0)), pi, gi))
        for iou, score, pi, gi in sorted(candidates, reverse=True):
            if pi not in used_pred and gi not in used_gt:
                used_pred.add(pi)
                used_gt.add(gi)
                counts['correct'] += 1

        # Remaining predictions: classification/localization/background/duplicate.
        for pi, p in enumerate(preds):
            if pi in used_pred:
                continue
            overlaps = [(iou_xywh(p['bbox'], g['bbox']), gi, g) for gi, g in enumerate(gts)]
            best_iou, best_gi, best_g = max(overlaps, default=(0.0, None, None), key=lambda x: x[0])
            pcls = int(p['category_id'])
            score = float(p.get('score', 0.0))

            if best_g is not None and best_iou >= iou_match and pcls != int(best_g['category_id']):
                etype = 'classification'
            elif best_g is not None and iou_loose <= best_iou < iou_match and pcls == int(best_g['category_id']):
                etype = 'localization'
            elif best_g is not None and best_iou >= iou_match and pcls == int(best_g['category_id']) and best_gi in used_gt:
                etype = 'duplicate'
            elif best_iou < iou_loose:
                etype = 'background_fp'
            else:
                # Overlaps an object but is both wrong class and poorly localized.
                etype = 'classification_localization'

            counts[etype] += 1
            errors.append({
                'image_id': image_id,
                'type': etype,
                'score': score,
                'pred_category_id': pcls,
                'pred_bbox': p['bbox'],
                'matched_gt_category_id': int(best_g['category_id']) if best_g else None,
                'matched_gt_bbox': best_g['bbox'] if best_g else None,
                'iou': float(best_iou),
            })

        # Remaining GTs are misses.
        for gi, g in enumerate(gts):
            if gi in used_gt:
                continue
            same_cls_best = 0.0
            wrong_cls_best = 0.0
            for p in preds:
                ov = iou_xywh(p['bbox'], g['bbox'])
                if int(p['category_id']) == int(g['category_id']):
                    same_cls_best = max(same_cls_best, ov)
                else:
                    wrong_cls_best = max(wrong_cls_best, ov)
            counts['missed_gt'] += 1
            errors.append({
                'image_id': image_id,
                'type': 'missed_gt',
                'score': None,
                'pred_category_id': None,
                'pred_bbox': None,
                'matched_gt_category_id': int(g['category_id']),
                'matched_gt_bbox': g['bbox'],
                'iou': float(max(same_cls_best, wrong_cls_best)),
            })

    return counts, errors


def select_examples(errors, limit_per_type=4):
    by_type = defaultdict(list)
    for e in errors:
        by_type[e['type']].append(e)

    # Most informative ordering: confident wrong predictions, then severe misses.
    selected = []
    priorities = ['classification', 'localization', 'background_fp', 'missed_gt', 'classification_localization', 'duplicate']
    for etype in priorities:
        items = by_type.get(etype, [])
        if etype in {'classification', 'background_fp', 'classification_localization', 'duplicate'}:
            items = sorted(items, key=lambda e: (e.get('score') or 0.0), reverse=True)
        elif etype == 'localization':
            items = sorted(items, key=lambda e: abs(0.5 - e.get('iou', 0.0)))
        elif etype == 'missed_gt':
            items = sorted(items, key=lambda e: e.get('iou', 0.0))
        selected.extend(items[:limit_per_type])
    return selected


def xywh_to_xyxy(b):
    x, y, w, h = map(float, b)
    return [x, y, x + w, y + h]


def draw_example(image_path: Path, out_path: Path, example: dict[str, Any], cat_names: dict[int, str]):
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    gt_bbox = example.get('matched_gt_bbox')
    pred_bbox = example.get('pred_bbox')
    if gt_bbox:
        box = xywh_to_xyxy(gt_bbox)
        draw.rectangle(box, outline='green', width=4)
        gt_id = example.get('matched_gt_category_id')
        draw.text((box[0], max(0, box[1]-14)), f"GT {cat_names.get(gt_id, str(gt_id))}", fill='green', font=font)
    if pred_bbox:
        box = xywh_to_xyxy(pred_bbox)
        draw.rectangle(box, outline='red', width=4)
        p_id = example.get('pred_category_id')
        score = example.get('score')
        label = f"PRED {cat_names.get(p_id, str(p_id))} {score:.3f}" if score is not None else f"PRED {cat_names.get(p_id, str(p_id))}"
        draw.text((box[0], min(img.height-14, box[3]+2)), label, fill='red', font=font)

    title = f"{example['type']} | IoU={example.get('iou', 0):.3f}"
    draw.rectangle([0, 0, min(img.width, 420), 20], fill='white')
    draw.text((4, 4), title, fill='black', font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--output-dir', default='results/failure_analysis')
    ap.add_argument('--conf', type=float, default=0.25, help='Minimum prediction score for qualitative failure analysis')
    ap.add_argument('--iou-match', type=float, default=0.5)
    ap.add_argument('--examples-per-type', type=int, default=4)
    args = ap.parse_args()

    root = resolve_project_root()
    gt_path = root / 'data' / 'processed' / f'instances_{args.split}.json'
    image_dir = root / 'data' / 'processed' / 'images' / args.split
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    pred_paths = {
        'E0_YOLOv8s_640': root / 'results' / 'test_e0' / 'coco_predictions.json',
        'E1_FasterRCNN': root / 'results' / 'test_e1' / 'coco_predictions.json',
        'E2_YOLOv8s_1280': root / 'results' / 'test_e2' / 'coco_predictions.json',
    }
    for name, p in pred_paths.items():
        if not p.is_file():
            raise FileNotFoundError(f'{name}: {p}')

    coco = load_json(gt_path)
    images = {int(x['id']): x for x in coco['images']}
    cat_names = {int(c['id']): str(c.get('name', c['id'])) for c in coco['categories']}
    gt_by_image = defaultdict(list)
    for a in coco['annotations']:
        gt_by_image[int(a['image_id'])].append(a)

    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summary = {}
    all_examples = {}

    csv_rows = []
    for model_name, pred_path in pred_paths.items():
        preds = [p for p in load_json(pred_path) if float(p.get('score', 0.0)) >= args.conf]
        pred_by_image = defaultdict(list)
        for p in preds:
            pred_by_image[int(p['image_id'])].append(p)

        counts, errors = classify_errors(gt_by_image, pred_by_image, iou_match=args.iou_match)
        selected = select_examples(errors, args.examples_per_type)
        all_summary[model_name] = {
            'prediction_count_after_conf_filter': len(preds),
            'confidence_threshold': args.conf,
            'iou_match': args.iou_match,
            'counts': dict(counts),
        }
        all_examples[model_name] = selected
        for k, v in sorted(counts.items()):
            csv_rows.append({'model': model_name, 'error_type': k, 'count': v})

        model_gallery = out_dir / 'gallery' / model_name
        for idx, ex in enumerate(selected, 1):
            meta = images[int(ex['image_id'])]
            img_path = image_dir / meta['file_name']
            ex['file_name'] = meta['file_name']
            if img_path.is_file():
                out_path = model_gallery / f"{idx:02d}_{ex['type']}_{Path(meta['file_name']).stem}.jpg"
                draw_example(img_path, out_path, ex, cat_names)
                ex['rendered_file'] = str(out_path.relative_to(root)).replace('\\', '/')

    with (out_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)
    with (out_dir / 'examples.json').open('w', encoding='utf-8') as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=2)
    with (out_dir / 'error_counts.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'error_type', 'count'])
        w.writeheader()
        w.writerows(csv_rows)

    print(json.dumps(all_summary, ensure_ascii=False, indent=2))
    print(f'Saved: {out_dir}')

if __name__ == '__main__':
    main()
