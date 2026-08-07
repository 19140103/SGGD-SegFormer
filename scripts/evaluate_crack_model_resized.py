"""Fast resized evaluation for cross-dataset crack segmentation.

This script is intended for DACL10K zero-shot testing, where original images can
be much larger than the SegCODEBRIM training crop. It resizes each image and mask
to a fixed evaluation size before inference and metric computation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
MMSEG_ROOT = BRIDGE_ROOT
if str(MMSEG_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSEG_ROOT))

from mmseg.apis import inference_model, init_model  # noqa: E402

from evaluate_crack_model_once import (  # noqa: E402
    EPS,
    counts_from_prediction,
    metrics_from_counts,
    model_size_summary,
    probs_from_result,
    read_mask,
    safe_cuda_synchronize,
    structure_metrics,
    threshold_sweep_row,
    try_complexity_summary,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a binary crack segmentation model at fixed size.")
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--data-root",
        default=str(BRIDGE_ROOT / "data" / "dacl10k_crack_mmseg_jpg"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--img-suffix", default=".jpg")
    parser.add_argument("--seg-map-suffix", default=".png")
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--skip-structure",
        action="store_true",
        help="Skip skeleton metrics for fast cross-dataset evaluation.")
    parser.add_argument("--sweep-start", type=float, default=0.01)
    parser.add_argument("--sweep-end", type=float, default=0.99)
    parser.add_argument("--sweep-step", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-complexity", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(BRIDGE_ROOT / "work_dirs" / "dacl10k_evaluation"))
    return parser.parse_args()


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    img_dir = data_root / "img_dir" / args.split
    ann_dir = data_root / "ann_dir" / args.split
    image_paths = sorted(img_dir.glob(f"*{args.img_suffix}"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    output_dir = Path(args.output_dir) / args.model_name / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    model = init_model(args.config, args.checkpoint, device=args.device)
    model.eval()
    size_summary = model_size_summary(model)

    thresholds = np.round(
        np.arange(args.sweep_start, args.sweep_end + args.sweep_step * 0.5,
                  args.sweep_step),
        decimals=6).astype(np.float32)
    sweep_tp = np.zeros_like(thresholds, dtype=np.float64)
    sweep_fp = np.zeros_like(thresholds, dtype=np.float64)
    total_pos = 0.0
    total_neg = 0.0

    global_tp = global_fp = global_fn = global_tn = 0
    global_skel_pred_inter_target = 0
    global_skel_pred_sum = 0
    global_skel_target_inter_pred = 0
    global_skel_target_sum = 0
    global_abs_skeleton_length_error = 0.0
    skeleton_length_valid_images = 0
    per_image_rows: list[dict] = []
    ois_f1_values: list[float] = []
    inference_times_ms: list[float] = []

    with torch.no_grad():
        for index, img_path in enumerate(image_paths, start=1):
            mask_path = ann_dir / f"{img_path.stem}{args.seg_map_suffix}"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {img_path.name}")

            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            image = cv2.resize(
                image, (args.eval_size, args.eval_size),
                interpolation=cv2.INTER_AREA)
            target = read_mask(mask_path).astype(np.uint8)
            target = cv2.resize(
                target, (args.eval_size, args.eval_size),
                interpolation=cv2.INTER_NEAREST).astype(bool)

            safe_cuda_synchronize(args.device)
            start = time.perf_counter()
            result = inference_model(model, image)
            safe_cuda_synchronize(args.device)
            inference_times_ms.append((time.perf_counter() - start) * 1000.0)

            probs = probs_from_result(result)
            if probs.shape != target.shape:
                probs = cv2.resize(
                    probs, (args.eval_size, args.eval_size),
                    interpolation=cv2.INTER_LINEAR)
            pred = probs >= args.threshold

            tp, fp, fn, tn = counts_from_prediction(pred, target)
            global_tp += tp
            global_fp += fp
            global_fn += fn
            global_tn += tn

            pos_probs = np.sort(probs[target])
            neg_probs = np.sort(probs[~target])
            pos_count = float(pos_probs.size)
            neg_count = float(neg_probs.size)
            total_pos += pos_count
            total_neg += neg_count
            img_tp = pos_count - np.searchsorted(
                pos_probs, thresholds, side="left")
            img_fp = neg_count - np.searchsorted(
                neg_probs, thresholds, side="left")
            sweep_tp += img_tp
            sweep_fp += img_fp
            img_fn = pos_count - img_tp
            img_f1 = (2 * img_tp) / np.maximum(
                2 * img_tp + img_fp + img_fn, EPS)
            ois_f1_values.append(float(np.max(img_f1)))

            pixel_metrics = metrics_from_counts(tp, fp, fn, tn)
            if args.skip_structure:
                struct = {
                    "cldice": None,
                    "pred_skeleton_length": None,
                    "target_skeleton_length": None,
                    "abs_skeleton_length_error": None,
                }
            else:
                struct = structure_metrics(pred, target)
                global_skel_pred_inter_target += struct[
                    "skel_pred_inter_target"]
                global_skel_pred_sum += struct["pred_skeleton_length"]
                global_skel_target_inter_pred += struct[
                    "skel_target_inter_pred"]
                global_skel_target_sum += struct["target_skeleton_length"]
                if struct["abs_skeleton_length_error"] is not None:
                    global_abs_skeleton_length_error += struct[
                        "abs_skeleton_length_error"]
                    skeleton_length_valid_images += 1

            per_image_rows.append({
                "image": img_path.name,
                "gt_pos_rate": float(target.mean()),
                "threshold": args.threshold,
                **pixel_metrics,
                "cldice": struct["cldice"],
                "pred_skeleton_length": struct["pred_skeleton_length"],
                "target_skeleton_length": struct["target_skeleton_length"],
                "abs_skeleton_length_error": struct[
                    "abs_skeleton_length_error"],
            })

            if index % 100 == 0 or index == len(image_paths):
                print(f"{args.model_name}: evaluated {index}/{len(image_paths)}",
                      flush=True)

    sweep_rows = [
        threshold_sweep_row(float(thr), float(tp), float(fp), total_pos,
                            total_neg)
        for thr, tp, fp in zip(thresholds, sweep_tp, sweep_fp)
    ]
    best_f1_row = max(sweep_rows, key=lambda row: row["f1"])
    best_miou_row = max(sweep_rows, key=lambda row: row["miou"])
    fixed_metrics = metrics_from_counts(global_tp, global_fp, global_fn,
                                        global_tn)

    if args.skip_structure:
        tprec = None
        tsens = None
        cldice = None
        mean_abs_skeleton_length_error = None
    else:
        tprec = global_skel_pred_inter_target / max(global_skel_pred_sum, EPS)
        tsens = global_skel_target_inter_pred / max(global_skel_target_sum, EPS)
        cldice = 2 * tprec * tsens / max(tprec + tsens, EPS)
        mean_abs_skeleton_length_error = (
            global_abs_skeleton_length_error /
            max(skeleton_length_valid_images, 1))

    summary = {
        "model_name": args.model_name,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "data_root": str(data_root),
        "split": args.split,
        "num_images": len(image_paths),
        "eval_size": args.eval_size,
        "threshold": args.threshold,
        "positive_pixel_rate": total_pos / max(total_pos + total_neg, EPS),
        "fixed_threshold": {
            **fixed_metrics,
            "tp": global_tp,
            "fp": global_fp,
            "fn": global_fn,
            "tn": global_tn,
            "cldice": cldice,
            "topological_precision": tprec,
            "topological_sensitivity": tsens,
            "mean_abs_skeleton_length_error": mean_abs_skeleton_length_error,
        },
        "best_f1_ods": best_f1_row,
        "best_miou": best_miou_row,
        "ois_f1": float(np.mean(ois_f1_values)),
        "size": size_summary,
        "efficiency": {
            "mean_inference_time_ms": float(np.mean(inference_times_ms)),
            "fps": float(1000.0 / max(np.mean(inference_times_ms), EPS)),
        },
        "complexity": {
            "flops": None,
            "flops_formatted": None,
            "complexity_params": None,
            "complexity_params_formatted": None,
            "complexity_error": "skipped",
        } if args.skip_complexity else try_complexity_summary(
            model, args.eval_size, args.eval_size),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_csv(output_dir / "threshold_sweep.csv", sweep_rows)
    write_rows(output_dir / "per_image_metrics.csv", per_image_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

