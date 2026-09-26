"""Prepare paired, nested DACL10K splits for the low-annotation study.

The script uses the fixed 587-image test split created by
``make_dacl10k_fewshot_split.py``. The remaining 388 images are split into
four positive-mask-density strata and one negative-image stratum. Three paired
repetitions are generated with nested 5%, 10% and 20% train/validation sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import cv2


RATIOS = {
    "05": {"total": 50, "positive": 16, "negative": 34},
    "10": {"total": 97, "positive": 28, "negative": 69},
    "20": {"total": 194, "positive": 56, "negative": 138},
}

REPEATS = [
    {"repeat": "seed01", "split_seed": 20260923, "training_seed": 31001},
    {"repeat": "seed02", "split_seed": 20260924, "training_seed": 31002},
    {"repeat": "seed03", "split_seed": 20260925, "training_seed": 31003},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create nested DACL10K train/validation splits.")
    parser.add_argument(
        "--source-root", type=Path, required=True,
        help="Converted 975-image DACL10K root containing validation images.")
    parser.add_argument(
        "--fixed-test-root", type=Path, required=True,
        help="The fixed split whose test directory contains 587 images.")
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Output root for generated data, manifests and experiment metadata.")
    return parser.parse_args()


def read_stems(path: Path, suffix: str) -> list[str]:
    stems = sorted(item.stem for item in path.glob(f"*{suffix}"))
    if not stems:
        raise FileNotFoundError(f"No {suffix} files found in {path}")
    return stems


def write_pair(source_root: Path, output_root: Path, split: str,
               stem: str) -> None:
    image_source = source_root / "img_dir" / "validation" / f"{stem}.jpg"
    mask_source = source_root / "ann_dir" / "validation" / f"{stem}.png"
    image_target = output_root / "img_dir" / split / image_source.name
    mask_target = output_root / "ann_dir" / split / mask_source.name
    image_target.parent.mkdir(parents=True, exist_ok=True)
    mask_target.parent.mkdir(parents=True, exist_ok=True)
    if not image_target.exists():
        os.link(image_source, image_target)

    source_mask = cv2.imread(str(mask_source), cv2.IMREAD_GRAYSCALE)
    if source_mask is None:
        raise FileNotFoundError(mask_source)
    binary_mask = (source_mask > 0).astype("uint8")
    if mask_target.exists():
        mask_target.unlink()
    if not cv2.imwrite(str(mask_target), binary_mask):
        raise OSError(f"Could not write {mask_target}")


def sha256_lines(lines: list[str]) -> str:
    payload = "".join(f"{line}\n" for line in sorted(lines)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_positive_bins(positive_rates: dict[str, float]) -> list[list[str]]:
    ordered = sorted(positive_rates, key=lambda stem: positive_rates[stem])
    bins: list[list[str]] = []
    for index in range(4):
        start = round(index * len(ordered) / 4)
        end = round((index + 1) * len(ordered) / 4)
        bins.append(ordered[start:end])
    if [len(group) for group in bins] != [28, 28, 28, 28]:
        raise RuntimeError(
            f"Unexpected positive-bin sizes: {[len(group) for group in bins]}")
    return bins


def split_candidates(positive_bins: list[list[str]], negatives: list[str],
                     seed: int) -> tuple[dict[str, list[str]],
                                         dict[str, list[str]]]:
    train = {"negative": [], "positive_bins": []}
    val = {"negative": [], "positive_bins": []}
    for index, original in enumerate(positive_bins):
        items = list(original)
        random.Random(seed + 101 * (index + 1)).shuffle(items)
        midpoint = len(items) // 2
        train["positive_bins"].append(items[:midpoint])
        val["positive_bins"].append(items[midpoint:])

    shuffled_negatives = list(negatives)
    random.Random(seed + 991).shuffle(shuffled_negatives)
    midpoint = len(shuffled_negatives) // 2
    train["negative"] = shuffled_negatives[:midpoint]
    val["negative"] = shuffled_negatives[midpoint:]
    return train, val


def nested_order(candidate: dict[str, list[str]], seed: int) -> dict[str, list[str]]:
    ordered_bins: list[list[str]] = []
    for index, original in enumerate(candidate["positive_bins"]):
        items = list(original)
        random.Random(seed + 2003 + index).shuffle(items)
        ordered_bins.append(items)
    negatives = list(candidate["negative"])
    random.Random(seed + 4001).shuffle(negatives)
    return {"positive_bins": ordered_bins, "negative": negatives}


def select_nested(candidate: dict[str, list[str]], positive_count: int,
                  negative_count: int) -> list[str]:
    if positive_count % 4:
        raise ValueError("Positive counts must be divisible by four density bins")
    per_bin = positive_count // 4
    positives = [stem for group in candidate["positive_bins"]
                 for stem in group[:per_bin]]
    return sorted(positives + candidate["negative"][:negative_count])


def main() -> None:
    args = parse_args()
    source_images = args.source_root / "img_dir" / "validation"
    source_masks = args.source_root / "ann_dir" / "validation"
    test_images = args.fixed_test_root / "img_dir" / "test"

    all_stems = read_stems(source_images, ".jpg")
    test_stems = read_stems(test_images, ".jpg")
    if len(all_stems) != 975 or len(test_stems) != 587:
        raise RuntimeError(
            f"Expected 975 source and 587 test images, got "
            f"{len(all_stems)} and {len(test_stems)}")
    if not set(test_stems).issubset(all_stems):
        raise RuntimeError("Fixed test set is not a subset of the source data")

    pool_stems = sorted(set(all_stems) - set(test_stems))
    if len(pool_stems) != 388:
        raise RuntimeError(
            f"Expected a 388-image sampling pool, got {len(pool_stems)}")

    positive_rates: dict[str, float] = {}
    negatives: list[str] = []
    for stem in pool_stems:
        mask_path = source_masks / f"{stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        rate = float((mask > 0).mean())
        if rate > 0:
            positive_rates[stem] = rate
        else:
            negatives.append(stem)
    if len(positive_rates) != 112 or len(negatives) != 276:
        raise RuntimeError(
            f"Expected 112 positive and 276 negative pool images, got "
            f"{len(positive_rates)} and {len(negatives)}")

    positive_bins = make_positive_bins(positive_rates)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "paired_nested_three_seed_dacl10k_variability",
        "source_root": str(args.source_root.resolve()),
        "fixed_test_root": str(args.fixed_test_root.resolve()),
        "source_images": len(all_stems),
        "fixed_test_images": len(test_stems),
        "sampling_pool_images": len(pool_stems),
        "sampling_pool_positive": len(positive_rates),
        "sampling_pool_negative": len(negatives),
        "fixed_test_sha256": sha256_lines(test_stems),
        "ratios": RATIOS,
        "repeats": [],
    }

    for repeat in REPEATS:
        repeat_name = repeat["repeat"]
        split_seed = repeat["split_seed"]
        train_candidate, val_candidate = split_candidates(
            positive_bins, negatives, split_seed)
        train_order = nested_order(train_candidate, split_seed + 5003)
        val_order = nested_order(val_candidate, split_seed + 7001)
        repeat_record = {**repeat, "settings": {}}
        previous_train: set[str] = set()
        previous_val: set[str] = set()

        for ratio, counts in RATIOS.items():
            train = select_nested(train_order, counts["positive"],
                                  counts["negative"])
            val = select_nested(val_order, counts["positive"],
                                counts["negative"])
            train_set = set(train)
            val_set = set(val)
            test_set = set(test_stems)
            if train_set & val_set or train_set & test_set or val_set & test_set:
                raise RuntimeError(f"Leakage detected in {repeat_name}/{ratio}")
            if previous_train and not previous_train.issubset(train_set):
                raise RuntimeError(
                    f"Training subsets are not nested in {repeat_name}")
            if previous_val and not previous_val.issubset(val_set):
                raise RuntimeError(
                    f"Validation subsets are not nested in {repeat_name}")
            previous_train, previous_val = train_set, val_set

            data_root = args.output_root / "data" / repeat_name / f"pct{ratio}"
            for stem in train:
                write_pair(args.source_root, data_root, "train", stem)
            for stem in val:
                write_pair(args.source_root, data_root, "val", stem)

            split_dir = args.output_root / "manifests" / repeat_name / f"pct{ratio}"
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / "train.txt").write_text(
                "".join(f"{stem}\n" for stem in train), encoding="utf-8")
            (split_dir / "val.txt").write_text(
                "".join(f"{stem}\n" for stem in val), encoding="utf-8")
            repeat_record["settings"][ratio] = {
                "data_root": str(data_root.resolve()),
                "train_images": len(train),
                "val_images": len(val),
                "train_positive": sum(stem in positive_rates for stem in train),
                "val_positive": sum(stem in positive_rates for stem in val),
                "train_sha256": sha256_lines(train),
                "val_sha256": sha256_lines(val),
            }
        manifest["repeats"].append(repeat_record)

    manifest_path = args.output_root / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
