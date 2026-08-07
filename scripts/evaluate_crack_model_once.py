"""One-pass evaluation for bridge crack segmentation papers.

This script is designed for the Scientific Reports route: run a trained model
once and save the metrics needed for the main table, efficiency table,
inspection-oriented structural metrics, sparsity analysis, and later
qualitative figures.

It supports both one-channel sigmoid heads and two-channel softmax heads.
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
from PIL import Image


BRIDGE_ROOT = Path(__file__).resolve().parents[1]
MMSEG_ROOT = BRIDGE_ROOT
if str(MMSEG_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSEG_ROOT))

from mmseg.apis import inference_model, init_model  # noqa: E402


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a binary crack segmentation model once.")
    parser.add_argument("config", help="MMSeg config path.")
    parser.add_argument("checkpoint", help="Checkpoint path.")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Readable model name used in output filenames.")
    parser.add_argument(
        "--data-root",
        default=str(BRIDGE_ROOT / "data" / "SegCODEBRIM_mmseg"),
        help="MMSeg-style dataset root with img_dir and ann_dir.")
    parser.add_argument("--split", default="test", help="Dataset split name.")
    parser.add_argument("--img-suffix", default=".png")
    parser.add_argument("--seg-map-suffix", default=".png")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fixed threshold for deployment-style metrics and saved masks.")
    parser.add_argument("--sweep-start", type=float, default=0.01)
    parser.add_argument("--sweep-end", type=float, default=0.99)
    parser.add_argument("--sweep-step", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        default=str(BRIDGE_ROOT / "work_dirs" / "paper_evaluation"),
        help="Directory for JSON/CSV outputs and optional predictions.")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save fixed-threshold masks and probability maps.")
    parser.add_argument(
        "--save-prob-npz",
        action="store_true",
        help="Save float32 probabilities as compressed NPZ files.")
    parser.add_argument(
        "--timing-warmup",
        type=int,
        default=5,
        help="Number of first images excluded from timing average.")
    parser.add_argument(
        "--skip-complexity",
        action="store_true",
        help="Skip FLOPs analysis if only metrics are needed.")
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def probs_from_result(result) -> np.ndarray:
    data = result.seg_logits.data.detach().float().cpu()
    if data.dim() == 3:
        if data.shape[0] == 1:
            probs = data.squeeze(0)
            if probs.min() < 0.0 or probs.max() > 1.0:
                probs = probs.sigmoid()
        elif data.shape[0] >= 2:
            probs = torch.softmax(data, dim=0)[1]
        else:
            raise ValueError(f"Invalid logit shape: {tuple(data.shape)}")
    elif data.dim() == 2:
        probs = data
        if probs.min() < 0.0 or probs.max() > 1.0:
            probs = probs.sigmoid()
    else:
        raise ValueError(f"Expected logits [C,H,W] or [H,W], got {tuple(data.shape)}")
    return probs.numpy().astype(np.float32, copy=False)


def counts_from_prediction(pred: np.ndarray,
                           target: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    tn = int(np.logical_and(~pred, ~target).sum())
    return tp, fp, fn, tn


def metrics_from_counts(tp: float, fp: float, fn: float, tn: float) -> dict:
    precision = tp / max(tp + fp, EPS)
    recall = tp / max(tp + fn, EPS)
    f1 = 2 * precision * recall / max(precision + recall, EPS)
    crack_iou = tp / max(tp + fp + fn, EPS)
    bg_iou = tn / max(tn + fp + fn, EPS)
    miou = 0.5 * (crack_iou + bg_iou)
    acc = (tp + tn) / max(tp + fp + fn + tn, EPS)
    fpr_bg = fp / max(fp + tn, EPS)
    pred_pos_rate = (tp + fp) / max(tp + fp + fn + tn, EPS)
    false_positive_area_rate = fp / max(tp + fp + fn + tn, EPS)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "crack_iou": crack_iou,
        "bg_iou": bg_iou,
        "miou": miou,
        "acc": acc,
        "false_positive_rate_bg": fpr_bg,
        "false_positive_area_rate": false_positive_area_rate,
        "pred_pos_rate": pred_pos_rate,
    }


def morph_skeleton(mask: np.ndarray) -> np.ndarray:
    img = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img) > 0:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = cv2.erode(img, element)
    return skeleton > 0


def component_count(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    num_labels, _ = cv2.connectedComponents(mask.astype(np.uint8), 8)
    return int(num_labels - 1)


def structure_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    pred_skel = morph_skeleton(pred)
    target_skel = morph_skeleton(target)
    pred_len = int(pred_skel.sum())
    target_len = int(target_skel.sum())
    skel_pred_inter_target = int(np.logical_and(pred_skel, target).sum())
    skel_target_inter_pred = int(np.logical_and(target_skel, pred).sum())
    tprec = skel_pred_inter_target / max(pred_len, EPS)
    tsens = skel_target_inter_pred / max(target_len, EPS)
    cldice = 2 * tprec * tsens / max(tprec + tsens, EPS)
    if target_len > 0:
        skeleton_length_error = (pred_len - target_len) / target_len
        abs_skeleton_length_error = abs(pred_len - target_len) / target_len
    else:
        skeleton_length_error = None
        abs_skeleton_length_error = None
    return {
        "cldice": cldice,
        "topological_precision": tprec,
        "topological_sensitivity": tsens,
        "pred_skeleton_length": pred_len,
        "target_skeleton_length": target_len,
        "skeleton_length_error": skeleton_length_error,
        "abs_skeleton_length_error": abs_skeleton_length_error,
        "pred_skeleton_components": component_count(pred_skel),
        "target_skeleton_components": component_count(target_skel),
        "skel_pred_inter_target": skel_pred_inter_target,
        "skel_target_inter_pred": skel_target_inter_pred,
    }


def safe_cuda_synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def model_size_summary(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params": total,
        "params_m": total / 1e6,
        "trainable_params": trainable,
        "trainable_params_m": trainable / 1e6,
    }


def try_complexity_summary(model: torch.nn.Module, height: int,
                           width: int) -> dict:
    try:
        from mmengine.analysis import get_model_complexity_info
        from mmengine.analysis.print_helper import _format_size
        from mmseg.structures import SegDataSample

        result_meta = {"ori_shape": (height, width), "pad_shape": (height, width)}
        data_batch = {
            "inputs": [torch.rand((3, height, width))],
            "data_samples": [SegDataSample(metainfo=result_meta)],
        }
        data = model.data_preprocessor(data_batch)
        old_aux = getattr(model, "auxiliary_head", None)
        had_aux = hasattr(model, "auxiliary_head")
        if had_aux:
            model.auxiliary_head = None
        outputs = get_model_complexity_info(
            model,
            input_shape=None,
            inputs=data["inputs"],
            show_table=False,
            show_arch=False)
        if had_aux:
            model.auxiliary_head = old_aux
        return {
            "flops": outputs.get("flops"),
            "flops_formatted": _format_size(outputs.get("flops", 0)),
            "complexity_params": outputs.get("params"),
            "complexity_params_formatted": _format_size(outputs.get("params", 0)),
            "complexity_error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "flops": None,
            "flops_formatted": None,
            "complexity_params": None,
            "complexity_params_formatted": None,
            "complexity_error": repr(exc),
        }


def threshold_sweep_row(threshold: float, tp: float, fp: float, total_pos: float,
                        total_neg: float) -> dict:
    fn = total_pos - tp
    tn = total_neg - fp
    row = {"threshold": float(threshold)}
    row.update(metrics_from_counts(tp, fp, fn, tn))
    row.update({"tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn)})
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_prediction_outputs(output_dir: Path, stem: str, probs: np.ndarray,
                            pred: np.ndarray, save_prob_npz: bool) -> None:
    mask_dir = output_dir / "masks"
    prob_png_dir = output_dir / "probability_png"
    mask_dir.mkdir(parents=True, exist_ok=True)
    prob_png_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((pred.astype(np.uint8) * 255)).save(mask_dir / f"{stem}.png")
    prob_u8 = np.clip(probs * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(prob_u8).save(prob_png_dir / f"{stem}.png")
    if save_prob_npz:
        prob_npz_dir = output_dir / "probability_npz"
        prob_npz_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prob_npz_dir / f"{stem}.npz", prob=probs)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)
    img_dir = data_root / "img_dir" / args.split
    ann_dir = data_root / "ann_dir" / args.split
    image_paths = sorted(img_dir.glob(f"*{args.img_suffix}"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    model_name = args.model_name or config_path.stem
    output_dir = Path(args.output_dir) / model_name / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = np.round(
        np.arange(args.sweep_start, args.sweep_end + args.sweep_step * 0.5,
                  args.sweep_step),
        decimals=6).astype(np.float32)
    sweep_tp = np.zeros_like(thresholds, dtype=np.float64)
    sweep_fp = np.zeros_like(thresholds, dtype=np.float64)
    total_pos = 0.0
    total_neg = 0.0

    model = init_model(str(config_path), str(checkpoint_path), device=args.device)
    model.eval()
    size_summary = model_size_summary(model)
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))

    global_tp = global_fp = global_fn = global_tn = 0
    global_skel_pred_inter_target = 0
    global_skel_pred_sum = 0
    global_skel_target_inter_pred = 0
    global_skel_target_sum = 0
    global_pred_skeleton_length = 0
    global_target_skeleton_length = 0
    global_abs_skeleton_length_error = 0.0
    skeleton_length_valid_images = 0
    ois_f1_values: list[float] = []
    per_image_rows: list[dict] = []
    inference_times_ms: list[float] = []
    input_height = input_width = None

    with torch.no_grad():
        for index, img_path in enumerate(image_paths, start=1):
            mask_path = ann_dir / f"{img_path.stem}{args.seg_map_suffix}"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for {img_path.name}: {mask_path}")

            target = read_mask(mask_path)
            input_height, input_width = target.shape

            safe_cuda_synchronize(args.device)
            start = time.perf_counter()
            result = inference_model(model, str(img_path))
            safe_cuda_synchronize(args.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if index > args.timing_warmup:
                inference_times_ms.append(elapsed_ms)

            probs = probs_from_result(result)
            if probs.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch for {img_path.name}: pred {probs.shape}, "
                    f"target {target.shape}")

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
            img_tp = pos_count - np.searchsorted(pos_probs, thresholds, side="left")
            img_fp = neg_count - np.searchsorted(neg_probs, thresholds, side="left")
            sweep_tp += img_tp
            sweep_fp += img_fp
            img_fn = pos_count - img_tp
            img_tn = neg_count - img_fp
            img_f1 = (2 * img_tp) / np.maximum(2 * img_tp + img_fp + img_fn, EPS)
            ois_f1_values.append(float(np.max(img_f1)))

            pixel_metrics = metrics_from_counts(tp, fp, fn, tn)
            struct = structure_metrics(pred, target)
            global_skel_pred_inter_target += struct["skel_pred_inter_target"]
            global_skel_pred_sum += struct["pred_skeleton_length"]
            global_skel_target_inter_pred += struct["skel_target_inter_pred"]
            global_skel_target_sum += struct["target_skeleton_length"]
            global_pred_skeleton_length += struct["pred_skeleton_length"]
            global_target_skeleton_length += struct["target_skeleton_length"]
            if struct["abs_skeleton_length_error"] is not None:
                global_abs_skeleton_length_error += struct[
                    "abs_skeleton_length_error"]
                skeleton_length_valid_images += 1

            row = {
                "image": img_path.name,
                "gt_pos_rate": float(target.mean()),
                "threshold": args.threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
            row.update(pixel_metrics)
            row.update({
                key: value
                for key, value in struct.items()
                if not key.startswith("skel_")
            })
            per_image_rows.append(row)

            if args.save_predictions:
                save_prediction_outputs(output_dir, img_path.stem, probs, pred,
                                        args.save_prob_npz)

            if index % 10 == 0 or index == len(image_paths):
                print(f"Evaluated {index}/{len(image_paths)} images", flush=True)

    sweep_rows = [
        threshold_sweep_row(thr, tp, fp, total_pos, total_neg)
        for thr, tp, fp in zip(thresholds, sweep_tp, sweep_fp)
    ]
    best_f1_row = max(sweep_rows, key=lambda row: row["f1"])
    best_miou_row = max(sweep_rows, key=lambda row: row["miou"])
    fixed_metrics = metrics_from_counts(global_tp, global_fp, global_fn, global_tn)
    tprec = global_skel_pred_inter_target / max(global_skel_pred_sum, EPS)
    tsens = global_skel_target_inter_pred / max(global_skel_target_sum, EPS)
    cldice = 2 * tprec * tsens / max(tprec + tsens, EPS)

    timing = {
        "timing_warmup": args.timing_warmup,
        "mean_inference_time_ms": float(np.mean(inference_times_ms))
        if inference_times_ms else None,
        "std_inference_time_ms": float(np.std(inference_times_ms))
        if inference_times_ms else None,
        "fps": float(1000.0 / np.mean(inference_times_ms))
        if inference_times_ms else None,
        "timed_images": len(inference_times_ms),
    }
    if args.device.startswith("cuda") and torch.cuda.is_available():
        timing["peak_cuda_memory_mb"] = float(
            torch.cuda.max_memory_allocated(torch.device(args.device)) /
            (1024**2))
    else:
        timing["peak_cuda_memory_mb"] = None

    if args.skip_complexity:
        complexity = {
            "flops": None,
            "flops_formatted": None,
            "complexity_params": None,
            "complexity_params_formatted": None,
            "complexity_error": "skipped",
        }
    else:
        complexity = try_complexity_summary(model, input_height, input_width)

    summary = {
        "model_name": model_name,
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "data_root": str(data_root.resolve()),
        "split": args.split,
        "num_images": len(image_paths),
        "input_shape_hw": [input_height, input_width],
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
            "pred_skeleton_length": global_pred_skeleton_length,
            "target_skeleton_length": global_target_skeleton_length,
            "skeleton_length_error": (
                (global_pred_skeleton_length - global_target_skeleton_length) /
                max(global_target_skeleton_length, EPS)),
            "mean_abs_skeleton_length_error": (
                global_abs_skeleton_length_error /
                max(skeleton_length_valid_images, 1)),
        },
        "best_f1_ods": best_f1_row,
        "best_miou": best_miou_row,
        "ois_f1": float(np.mean(ois_f1_values)),
        "size": size_summary,
        "efficiency": timing,
        "complexity": complexity,
        "prediction_outputs_saved": bool(args.save_predictions),
    }

    summary_path = output_dir / "summary.json"
    per_image_path = output_dir / "per_image_metrics.csv"
    sweep_path = output_dir / "threshold_sweep.csv"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_csv(per_image_path, per_image_rows)
    write_csv(sweep_path, sweep_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to {summary_path}")
    print(f"Saved per-image metrics to {per_image_path}")
    print(f"Saved threshold sweep to {sweep_path}")


if __name__ == "__main__":
    main()

