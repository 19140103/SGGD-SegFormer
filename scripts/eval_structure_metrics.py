"""Evaluate binary crack segmentation with skeleton continuity metrics."""

from __future__ import annotations

import argparse
import json
import sys
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
        description="Evaluate crack F1/IoU and hard clDice at one threshold.")
    parser.add_argument("config", help="MMSeg config path.")
    parser.add_argument("checkpoint", help="Checkpoint path.")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument(
        "--data-root",
        default=str(BRIDGE_ROOT / "data" / "SegCODEBRIM_mmseg"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        default=str(BRIDGE_ROOT / "work_dirs" / "structure_metrics"))
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
    elif data.dim() != 2:
        raise ValueError(f"Expected binary logits with shape [1,H,W], got {data.shape}")
    probs = data.cpu().numpy()
    if probs.min() < 0.0 or probs.max() > 1.0:
        probs = 1.0 / (1.0 + np.exp(-probs))
    return probs.astype(np.float32, copy=False)


def morph_skeleton(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton for binary masks using OpenCV primitives."""
    img = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img) > 0:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = cv2.erode(img, element)
    return skeleton > 0


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    img_dir = data_root / "img_dir" / args.split
    ann_dir = data_root / "ann_dir" / args.split
    image_paths = sorted(img_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    model = init_model(args.config, args.checkpoint, device=args.device)
    model.eval()

    tp = fp = fn = tn = 0
    skel_pred_inter_target = 0
    skel_pred_sum = 0
    skel_target_inter_pred = 0
    skel_target_sum = 0

    with torch.no_grad():
        for index, img_path in enumerate(image_paths, start=1):
            target = read_mask(ann_dir / f"{img_path.stem}.png")
            probs = probs_from_result(inference_model(model, str(img_path)))
            pred = probs >= args.threshold
            if pred.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch for {img_path.name}: pred {pred.shape}, "
                    f"target {target.shape}")

            tp += int(np.logical_and(pred, target).sum())
            fp += int(np.logical_and(pred, ~target).sum())
            fn += int(np.logical_and(~pred, target).sum())
            tn += int(np.logical_and(~pred, ~target).sum())

            pred_skel = morph_skeleton(pred)
            target_skel = morph_skeleton(target)
            skel_pred_inter_target += int(np.logical_and(pred_skel, target).sum())
            skel_pred_sum += int(pred_skel.sum())
            skel_target_inter_pred += int(np.logical_and(target_skel, pred).sum())
            skel_target_sum += int(target_skel.sum())

            if index % 10 == 0 or index == len(image_paths):
                print(f"Evaluated {index}/{len(image_paths)} images", flush=True)

    precision = tp / max(tp + fp, EPS)
    recall = tp / max(tp + fn, EPS)
    f1 = 2 * precision * recall / max(precision + recall, EPS)
    crack_iou = tp / max(tp + fp + fn, EPS)
    bg_iou = tn / max(tn + fp + fn, EPS)
    miou = 0.5 * (crack_iou + bg_iou)
    tprec = skel_pred_inter_target / max(skel_pred_sum, EPS)
    tsens = skel_target_inter_pred / max(skel_target_sum, EPS)
    cldice = 2 * tprec * tsens / max(tprec + tsens, EPS)
    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "threshold": args.threshold,
        "num_images": len(image_paths),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "crack_iou": crack_iou,
        "bg_iou": bg_iou,
        "miou": miou,
        "cldice": cldice,
        "topological_precision": tprec,
        "topological_sensitivity": tsens,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (
        f"{Path(args.config).stem}_{args.split}_"
        f"{Path(args.checkpoint).stem}_thr{args.threshold:.2f}.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()

