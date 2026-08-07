"""Convert DACL10K Crack/ACrack polygons to MMSeg binary masks.

Expected raw layout follows dacl10k-toolkit:

source_root/
  images/<split>/*.jpg
  annotations/<split>/*.json

The output layout matches the SegCODEBRIM MMSeg folder:

output_root/
  img_dir/<split>/*.png
  ann_dir/<split>/*.png
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


BRIDGE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a binary DACL10K crack subset for evaluation.")
    parser.add_argument(
        "--source-root",
        default=str(BRIDGE_ROOT / "data" / "dacl10k"),
        help="Raw DACL10K root containing images/ and annotations/.")
    parser.add_argument(
        "--output-root",
        default=str(BRIDGE_ROOT / "data" / "dacl10k_crack_mmseg"),
        help="Output MMSeg-style root.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["validation"],
        help="Splits to convert, e.g. train validation test.")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["Crack", "ACrack"],
        help="DACL10K labels merged into the binary crack mask.")
    parser.add_argument(
        "--copy-jpg",
        action="store_true",
        help="Copy original images instead of converting them to PNG.")
    return parser.parse_args()


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        path = images_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    matches = list(images_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def mask_from_annotation(annotation_path: Path,
                         target_labels: set[str]) -> tuple[Image.Image, dict]:
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    width = int(data["imageWidth"])
    height = int(data["imageHeight"])
    mask = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(mask)
    label_counts = {label: 0 for label in sorted(target_labels)}
    for shape in data.get("shapes", []):
        label = shape.get("label")
        points = shape.get("points", [])
        if label not in target_labels or len(points) < 2:
            continue
        xy = [(float(x), float(y)) for x, y in points]
        if len(xy) >= 3:
            drawer.polygon(xy, outline=255, fill=255)
        else:
            drawer.line(xy, fill=255, width=3)
        label_counts[label] = label_counts.get(label, 0) + 1
    return mask, label_counts


def save_image(src: Path, dst: Path, copy_jpg: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_jpg and src.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        shutil.copy2(src, dst.with_suffix(src.suffix.lower()))
        return
    with Image.open(src) as img:
        img.convert("RGB").save(dst.with_suffix(".png"))


def convert_split(source_root: Path, output_root: Path, split: str,
                  labels: set[str], copy_jpg: bool) -> dict:
    images_dir = source_root / "images" / split
    annotations_dir = source_root / "annotations" / split
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Missing annotation split: {annotations_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing image split: {images_dir}")

    out_img_dir = output_root / "img_dir" / split
    out_ann_dir = output_root / "ann_dir" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    annotation_paths = sorted(annotations_dir.glob("*.json"))
    if not annotation_paths:
        raise FileNotFoundError(f"No JSON annotations found in {annotations_dir}")

    total_pixels = 0
    positive_pixels = 0
    converted = 0
    missing_images = []
    positive_images = 0
    label_counts_total = {label: 0 for label in sorted(labels)}

    for index, ann_path in enumerate(annotation_paths, start=1):
        stem = ann_path.stem
        image_path = find_image(images_dir, stem)
        if image_path is None:
            missing_images.append(stem)
            continue

        mask, label_counts = mask_from_annotation(ann_path, labels)
        for label, count in label_counts.items():
            label_counts_total[label] = label_counts_total.get(label, 0) + count
        mask_arr = np.asarray(mask) > 0
        total_pixels += int(mask_arr.size)
        positive_pixels += int(mask_arr.sum())
        if mask_arr.any():
            positive_images += 1

        save_image(image_path, out_img_dir / f"{stem}.png", copy_jpg=copy_jpg)
        mask.save(out_ann_dir / f"{stem}.png")
        converted += 1
        if index % 250 == 0 or index == len(annotation_paths):
            print(
                f"{split}: converted {index}/{len(annotation_paths)} annotations",
                flush=True)

    return {
        "split": split,
        "converted_images": converted,
        "missing_images": missing_images,
        "positive_images": positive_images,
        "positive_image_rate": positive_images / max(converted, 1),
        "positive_pixels": positive_pixels,
        "total_pixels": total_pixels,
        "positive_pixel_rate": positive_pixels / max(total_pixels, 1),
        "label_counts": label_counts_total,
    }


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    labels = set(args.labels)
    summaries = []
    for split in args.splits:
        summaries.append(
            convert_split(source_root, output_root, split, labels,
                          args.copy_jpg))
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "conversion_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_root": str(source_root.resolve()),
                "output_root": str(output_root.resolve()),
                "labels": sorted(labels),
                "splits": summaries,
            },
            f,
            ensure_ascii=False,
            indent=2)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Saved conversion summary to {summary_path}")


if __name__ == "__main__":
    main()
