from __future__ import annotations

from mmengine.registry import init_default_scope
from mmseg.registry import DATASETS
import mmseg.datasets  # noqa: F401


def main() -> None:
    init_default_scope("mmseg")
    cfg = dict(
        type="BaseSegDataset",
        data_root="data/SegCODEBRIM_mmseg",
        data_prefix=dict(
            img_path="img_dir/train",
            seg_map_path="ann_dir/train",
        ),
        img_suffix=".png",
        seg_map_suffix=".png",
        metainfo=dict(
            classes=("background", "crack"),
            palette=[[0, 0, 0], [255, 255, 255]],
        ),
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="PackSegInputs"),
        ],
    )
    dataset = DATASETS.build(cfg)
    sample = dataset[0]
    gt = sample["data_samples"].gt_sem_seg.data
    print(f"len={len(dataset)}")
    print(f"inputs_shape={tuple(sample['inputs'].shape)}")
    print(f"gt_shape={tuple(gt.shape)}")
    print(f"gt_values={gt.unique().tolist()}")


if __name__ == "__main__":
    main()

