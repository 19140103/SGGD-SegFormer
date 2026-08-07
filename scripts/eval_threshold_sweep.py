"""Threshold-sweep evaluation for sparse binary crack segmentation.

This script evaluates MMSegmentation models with one-channel binary heads on
SegCODEBRIM-style folders. It reports both fixed-threshold metrics and ODS/OIS
style best-threshold scores commonly used by crack segmentation papers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
MMSEG_ROOT = BRIDGE_ROOT
if str(MMSEG_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSEG_ROOT))

from mmseg.apis import inference_model, init_model  # noqa: E402


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep thresholds for binary crack segmentation metrics.")
    parser.add_argument("config", help="MMSeg config path.")
    parser.add_argument("checkpoint", help="Checkpoint path.")
    parser.add_argument(
        "--data-root",
        default=str(BRIDGE_ROOT / "data" / "SegCODEBRIM_mmseg"),
        help="Dataset root with img_dir and ann_dir.")
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.")
    parser.add_argument(
        "--device", default="cuda:0", help="Device passed to init_model.")
    parser.add_argument(
        "--start", type=float, default=0.01, help="First threshold.")
    parser.add_argument(
        "--end", type=float, default=0.99, help="Last threshold.")
    parser.add_argument(
        "--step", type=float, default=0.01, help="Threshold step.")
    parser.add_argument(
        "--output-dir",
        default=str(BRIDGE_ROOT / "work_dirs" / "threshold_sweeps"),
        help="Directory for CSV/JSON outputs.")
    parser.add_argument(
        "--img-suffix", default=".png", help="Image filename suffix.")
    parser.add_argument(
        "--seg-map-suffix", default=".png", help="Mask filename suffix.")
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def probs_from_result(result) -> np.ndarray:
    data = result.seg_logits.data.detach().float()
    if data.dim() == 3 and data.shape[0] == 1:
        data = data.squeeze(0)
    elif data.dim() == 2:
        pass
    else:
        raise ValueError(f"Expected binary logits with shape [1,H,W], got {data.shape}")

    probs = data.cpu().numpy()
    if probs.min() < 0.0 or probs.max() > 1.0:
        probs = 1.0 / (1.0 + np.exp(-probs))
    return probs.astype(np.float32, copy=False)


def metrics_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray,
                        tn: np.ndarray) -> dict[str, np.ndarray]:
    precision = tp / np.maximum(tp + fp, EPS)
    recall = tp / np.maximum(tp + fn, EPS)
    f1 = 2 * precision * recall / np.maximum(precision + recall, EPS)
    crack_iou = tp / np.maximum(tp + fp + fn, EPS)
    bg_iou = tn / np.maximum(tn + fp + fn, EPS)
    miou = 0.5 * (crack_iou + bg_iou)
    acc = (tp + tn) / np.maximum(tp + fp + fn + tn, EPS)
    fpr = fp / np.maximum(fp + tn, EPS)
    pred_pos_rate = (tp + fp) / np.maximum(tp + fp + fn + tn, EPS)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "crack_iou": crack_iou,
        "bg_iou": bg_iou,
        "miou": miou,
        "acc": acc,
        "fpr": fpr,
        "pred_pos_rate": pred_pos_rate,
    }


def scalar_row(threshold: float, counts: tuple[float, float, float, float]) -> dict:
    tp, fp, fn, tn = [np.asarray([x], dtype=np.float64) for x in counts]
    metrics = metrics_from_counts(tp, fp, fn, tn)
    row = {"threshold": float(threshold)}
    row.update({key: float(value[0]) for key, value in metrics.items()})
    row.update({
        "tp": float(tp[0]),
        "fp": float(fp[0]),
        "fn": float(fn[0]),
        "tn": float(tn[0]),
    })
    return row


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    img_dir = data_root / "img_dir" / args.split
    ann_dir = data_root / "ann_dir" / args.split
    image_paths = sorted(img_dir.glob(f"*{args.img_suffix}"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    thresholds = np.round(
        np.arange(args.start, args.end + args.step * 0.5, args.step),
        decimals=6).astype(np.float32)
    tp = np.zeros_like(thresholds, dtype=np.float64)
    fp = np.zeros_like(thresholds, dtype=np.float64)
    total_pos = 0.0
    total_neg = 0.0
    ois_f1_values: list[float] = []

    model = init_model(args.config, args.checkpoint, device=args.device)
    model.eval()

    with torch.no_grad():
        for index, img_path in enumerate(image_paths, start=1):
            mask_path = ann_dir / f"{img_path.stem}{args.seg_map_suffix}"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {img_path.name}: {mask_path}")

            target = read_mask(mask_path)
            result = inference_model(model, str(img_path))
            probs = probs_from_result(result)
            if probs.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch for {img_path.name}: pred {probs.shape}, "
                    f"target {target.shape}")

            pos_probs = np.sort(probs[target])
            neg_probs = np.sort(probs[~target])
            pos_count = float(pos_probs.size)
            neg_count = float(neg_probs.size)
            total_pos += pos_count
            total_neg += neg_count

            img_tp = pos_count - np.searchsorted(pos_probs, thresholds, side="left")
            img_fp = neg_count - np.searchsorted(neg_probs, thresholds, side="left")
            tp += img_tp
            fp += img_fp

            img_fn = pos_count - img_tp
            img_tn = neg_count - img_fp
            img_metrics = metrics_from_counts(img_tp, img_fp, img_fn, img_tn)
            ois_f1_values.append(float(np.max(img_metrics["f1"])))

            if index % 10 == 0 or index == len(image_paths):
                print(f"Evaluated {index}/{len(image_paths)} images", flush=True)

    fn = total_pos - tp
    tn = total_neg - fp
    metrics = metrics_from_counts(tp, fp, fn, tn)

    rows = []
    for i, threshold in enumerate(thresholds):
        rows.append({
            "threshold": float(threshold),
            "precision": float(metrics["precision"][i]),
            "recall": float(metrics["recall"][i]),
            "f1": float(metrics["f1"][i]),
            "crack_iou": float(metrics["crack_iou"][i]),
            "bg_iou": float(metrics["bg_iou"][i]),
            "miou": float(metrics["miou"][i]),
            "acc": float(metrics["acc"][i]),
            "fpr": float(metrics["fpr"][i]),
            "pred_pos_rate": float(metrics["pred_pos_rate"][i]),
            "tp": float(tp[i]),
            "fp": float(fp[i]),
            "fn": float(fn[i]),
            "tn": float(tn[i]),
        })

    best_f1_row = max(rows, key=lambda row: row["f1"])
    best_miou_row = max(rows, key=lambda row: row["miou"])
    fixed_idx = int(np.argmin(np.abs(thresholds - 0.5)))
    fixed_row = rows[fixed_idx]
    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_root": str(data_root.resolve()),
        "split": args.split,
        "num_images": len(image_paths),
        "positive_pixel_rate": total_pos / max(total_pos + total_neg, EPS),
        "fixed_threshold_0.5": fixed_row,
        "best_f1_ods": best_f1_row,
        "best_miou": best_miou_row,
        "ois_f1": float(np.mean(ois_f1_values)),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.config).stem}_{args.split}_{Path(args.checkpoint).stem}"
    csv_path = output_dir / f"{stem}_thresholds.csv"
    json_path = output_dir / f"{stem}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved threshold table to {csv_path}")
    print(f"Saved summary to {json_path}")


if __name__ == "__main__":
    main()

