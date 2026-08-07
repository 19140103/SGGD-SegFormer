from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def group_key(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 4 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2])
    return path.stem


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_mask(src: Path, dst: Path) -> None:
    arr = np.array(Image.open(src))
    if arr.ndim == 3:
        arr = arr[..., 0]
    out = (arr > 0).astype(np.uint8)
    Image.fromarray(out, mode="L").save(dst)


def split_groups(groups: list[str], seed: int) -> dict[str, set[str]]:
    rng = random.Random(seed)
    groups = list(groups)
    rng.shuffle(groups)
    n = len(groups)
    n_train = round(n * 0.7)
    n_val = round(n * 0.15)
    return {
        "train": set(groups[:n_train]),
        "val": set(groups[n_train : n_train + n_val]),
        "test": set(groups[n_train + n_val :]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    img_root = args.raw_root / "images"
    gt_root = args.raw_root / "gts"
    if not img_root.is_dir() or not gt_root.is_dir():
        raise FileNotFoundError("Expected raw SegCODEBRIM/images and gts directories.")

    images = sorted(img_root.glob("*.png"))
    if not images:
        raise RuntimeError(f"No PNG images found in {img_root}")

    missing = [p.name for p in images if not (gt_root / p.name).exists()]
    if missing:
        raise RuntimeError(f"Missing masks for {len(missing)} images, e.g. {missing[:5]}")

    all_groups = sorted({group_key(p) for p in images})
    splits = split_groups(all_groups, args.seed)

    for split in ("train", "val", "test"):
        (args.out_root / "img_dir" / split).mkdir(parents=True, exist_ok=True)
        (args.out_root / "ann_dir" / split).mkdir(parents=True, exist_ok=True)
        (args.out_root / "splits").mkdir(parents=True, exist_ok=True)

    counts = {split: 0 for split in splits}
    group_counts = {split: len(splits[split]) for split in splits}
    split_stems = {split: [] for split in splits}

    for img_path in images:
        key = group_key(img_path)
        split = next(name for name, keys in splits.items() if key in keys)
        stem = img_path.stem
        split_stems[split].append(stem)
        counts[split] += 1

        link_or_copy(img_path, args.out_root / "img_dir" / split / img_path.name)
        write_mask(gt_root / img_path.name, args.out_root / "ann_dir" / split / img_path.name)

    for split, stems in split_stems.items():
        with (args.out_root / "splits" / f"{split}.txt").open("w", encoding="utf-8") as f:
            f.write("\n".join(sorted(stems)))
            f.write("\n")

    print(f"raw_root={args.raw_root}")
    print(f"out_root={args.out_root}")
    print(f"seed={args.seed}")
    print(f"groups={len(all_groups)}")
    for split in ("train", "val", "test"):
        print(f"{split}: groups={group_counts[split]}, images={counts[split]}")


if __name__ == "__main__":
    main()
