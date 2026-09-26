"""Create a stratified DACL10K Crack/ACrack split for MMSeg.

The converted source dataset contains only binary crack masks generated from
the DACL10K Crack and ACrack classes. This script keeps that binary setting and
creates a train/val/test starting split. For the repeated low-annotation
protocol used in the manuscript, first create the fixed 587-image test split
with 20% train and 20% validation settings, then use
``make_dacl10k_nested_splits.py`` to generate nested repetitions from its
388-image pool.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


BRIDGE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default=str(BRIDGE_ROOT / "data" / "dacl10k_crack_mmseg_jpg"))
    parser.add_argument(
        "--output-root",
        default=str(BRIDGE_ROOT / "data" / "dacl10k_crack_fewshot10_mmseg"))
    parser.add_argument("--source-split", default="validation")
    parser.add_argument("--train-ratio", type=float, default=0.10)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--density-bins",
        type=int,
        default=4,
        help=(
            "Number of positive-mask density bins used for stratification. "
            "Negative images are stratified as a separate group."))
    return parser.parse_args()


def copy_pair(source_root: Path, output_root: Path, source_split: str,
              target_split: str, stem: str) -> None:
    image_src = source_root / "img_dir" / source_split / f"{stem}.jpg"
    mask_src = source_root / "ann_dir" / source_split / f"{stem}.png"
    image_dst = output_root / "img_dir" / target_split / image_src.name
    mask_dst = output_root / "ann_dir" / target_split / mask_src.name
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    mask_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_src, image_dst)
    mask = cv2.imread(str(mask_src), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_src)
    mask = (mask > 0).astype(np.uint8)
    cv2.imwrite(str(mask_dst), mask)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    source_split = args.source_split

    image_dir = source_root / "img_dir" / source_split
    mask_dir = source_root / "ann_dir" / source_split
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(image_dir)

    positives: list[tuple[str, float]] = []
    negatives: list[str] = []
    for image_path in image_paths:
        mask = cv2.imread(str(mask_dir / f"{image_path.stem}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_dir / f"{image_path.stem}.png")
        positive_rate = float((mask > 0).mean())
        if positive_rate > 0:
            positives.append((image_path.stem, positive_rate))
        else:
            negatives.append(image_path.stem)

    rng = random.Random(args.seed)
    rng.shuffle(negatives)

    def partition(items: list[str]) -> tuple[list[str], list[str], list[str]]:
        if not items:
            return [], [], []
        train_count = max(1, round(len(items) * args.train_ratio))
        val_count = max(1, round(len(items) * args.val_ratio))
        train = items[:train_count]
        val = items[train_count:train_count + val_count]
        test = items[train_count + val_count:]
        return train, val, test

    positive_bins: list[list[str]] = []
    positives_sorted = sorted(positives, key=lambda item: item[1])
    bin_count = max(1, min(args.density_bins, len(positives_sorted)))
    for bin_index in range(bin_count):
        start = round(bin_index * len(positives_sorted) / bin_count)
        end = round((bin_index + 1) * len(positives_sorted) / bin_count)
        bin_items = [stem for stem, _ in positives_sorted[start:end]]
        rng.shuffle(bin_items)
        positive_bins.append(bin_items)

    pos_train: list[str] = []
    pos_val: list[str] = []
    pos_test: list[str] = []
    for bin_items in positive_bins:
        train, val, test = partition(bin_items)
        pos_train.extend(train)
        pos_val.extend(val)
        pos_test.extend(test)

    neg_train, neg_val, neg_test = partition(negatives)
    splits = {
        "train": sorted(pos_train + neg_train),
        "val": sorted(pos_val + neg_val),
        "test": sorted(pos_test + neg_test),
    }

    if output_root.exists():
        shutil.rmtree(output_root)
    for split, stems in splits.items():
        for stem in stems:
            copy_pair(source_root, output_root, source_split, split, stem)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_split": source_split,
        "labels": ["ACrack", "Crack"],
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "splits": {},
    }
    positive_rates = dict(positives)
    positive_set = set(positive_rates)
    for split, stems in splits.items():
        pos_count = sum(stem in positive_set for stem in stems)
        density_values = [positive_rates[stem] for stem in stems
                          if stem in positive_set]
        summary["splits"][split] = {
            "images": len(stems),
            "positive_images": pos_count,
            "negative_images": len(stems) - pos_count,
            "positive_image_rate": pos_count / max(len(stems), 1),
            "mean_positive_mask_density": (
                float(np.mean(density_values)) if density_values else 0.0),
            "median_positive_mask_density": (
                float(np.median(density_values)) if density_values else 0.0),
        }

    with (output_root / "fewshot_summary.json").open("w",
                                                     encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
