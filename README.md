# SGGD-SegFormer

Code release for **SGGD: A Structure-Guided Gated Decoder for Lightweight Concrete Bridge Crack Segmentation in Remote Inspection Imagery**.

This repository is a lightweight MMSegmentation-based project for binary bridge crack segmentation. It includes the SGGD decoder implementation, bridge crack experiment configs, dataset preparation utilities, and evaluation scripts.

## What is included

- SGGD decoder implementation:
  - `mmseg/models/decode_heads/spg_segformer_head.py`
  - public config class name: `StructureGuidedGatedDecoderHead`
- Binary crack losses:
  - `BinaryDiceLoss`
  - `BinaryTverskyLoss`
  - `SoftClDiceLoss`
- SegCODEBRIM and DACL10K bridge crack configs:
  - `configs/bridge_crack/`
- Dataset preparation and evaluation scripts:
  - `scripts/prepare_segcodebrim_mmseg.py`
  - `scripts/prepare_dacl10k_crack_mmseg.py`
  - `scripts/make_dacl10k_fewshot_split.py`
  - `scripts/evaluate_crack_model_once.py`
  - `scripts/evaluate_crack_model_resized.py`
  - `scripts/eval_threshold_sweep.py`
  - `scripts/eval_structure_metrics.py`
  - `scripts/verify_mmseg_dataset.py`

## Installation

Create an environment following the MMSegmentation requirements, then install this project in editable mode:

```bash
pip install -U openmim
mim install "mmengine>=0.7.0" "mmcv>=2.0.0"
pip install -e .
```

Additional packages used by the crack preparation and evaluation scripts can be installed with:

```bash
pip install -r requirements/crack.txt
```

## Dataset Layout

The default configs use relative paths under `data/`:

```text
data/
  SegCODEBRIM_mmseg/
    img_dir/{train,val,test}/
    ann_dir/{train,val,test}/
  dacl10k_crack_fewshot10_mmseg/
    img_dir/{train,val,test}/
    ann_dir/{train,val,test}/
```

The raw datasets are obtained from their official sources and are not bundled with this repository. Prepare the converted datasets with:

```bash
python scripts/prepare_segcodebrim_mmseg.py --raw-root /path/to/SegCODEBRIM --out-root data/SegCODEBRIM_mmseg
python scripts/prepare_dacl10k_crack_mmseg.py --source-root /path/to/dacl10k --output-root data/dacl10k_crack_mmseg_jpg --splits validation --copy-jpg
```

The DACL10K conversion merges the `Crack` and `ACrack` annotations into one
binary crack class. The processed subset used in the manuscript contains 975
images. It has a fixed 587-image test set and a 388-image sampling pool, with
112 positive and 276 negative images in that pool.

The fixed-test starting split can be created from the converted validation
split with:

```bash
python scripts/make_dacl10k_fewshot_split.py \
  --source-root data/dacl10k_crack_mmseg_jpg \
  --output-root data/dacl10k_crack_fewshot20_mmseg \
  --train-ratio 0.20 --val-ratio 0.20 --seed 20260804
```

For the repeated low-annotation protocol reported in the manuscript, pass the
converted source root and this fixed-test split to the nested split generator:

```bash
python scripts/make_dacl10k_nested_splits.py \
  --source-root data/dacl10k_crack_mmseg_jpg \
  --fixed-test-root data/dacl10k_crack_fewshot20_mmseg \
  --output-root data/dacl10k_repeated_protocol
```

This creates three paired repetitions with split seeds `20260923`, `20260924`
and `20260925`. The corresponding fine-tuning seeds are `31001`, `31002` and
`31003`. Each repetition uses the same 587-image test set and nested
training/validation subsets of 50/50, 97/97 and 194/194 images. The generated
data/ tree contains the train/validation roots, while the fixed-test root is
used separately for final evaluation. Split manifests and an
experiment_manifest.json are written alongside the generated data.

## Training

Train the main SGGD-SegFormer model:

```bash
python tools/train.py configs/bridge_crack/sggd_segformer_b0_segcodebrim_512x512_binary_bce_tversky.py
```

Train the SegFormer-B0 baseline:

```bash
python tools/train.py configs/bridge_crack/segformer_b0_segcodebrim_512x512_binary_bce_tversky.py
```

DACL10K low-annotation configs set `load_from = None` by default. To fine-tune from a SegCODEBRIM checkpoint, pass the checkpoint path explicitly:

```bash
python tools/train.py configs/bridge_crack/sggd_segformer_b0_dacl10k_fewshot10_512x512_bce_tversky.py --cfg-options load_from=/path/to/segcodebrim_checkpoint.pth
```

Pretrained checkpoints are not committed to this source repository. If the
paper is accepted, the default SGGD-SegFormer checkpoint and the matched
SegFormer-B0 baseline checkpoint will be released through a GitHub Release or
an archival repository, together with the corresponding configs and checksums.
The source code and configs remain usable without the checkpoints by training
from the documented initialization settings.

## Evaluation

Run one-pass evaluation with ODS threshold sweeping, fixed-threshold metrics, complexity, and FPS:

```bash
python scripts/evaluate_crack_model_once.py configs/bridge_crack/sggd_segformer_b0_segcodebrim_512x512_binary_bce_tversky.py /path/to/checkpoint.pth --data-root data/SegCODEBRIM_mmseg --split test
```

Run threshold sweeping only:

```bash
python scripts/eval_threshold_sweep.py configs/bridge_crack/sggd_segformer_b0_segcodebrim_512x512_binary_bce_tversky.py /path/to/checkpoint.pth --data-root data/SegCODEBRIM_mmseg --split test
```

## Attribution and License

This project is built on MMSegmentation and is released under the Apache License 2.0. The original MMSegmentation code is copyrighted by the MMSegmentation Authors. The SGGD decoder, bridge crack experiment configs, dataset preparation utilities, and crack evaluation scripts are provided as additions in this repository.

## Notes

The decoder source file keeps the historical filename `spg_segformer_head.py` for checkpoint compatibility. The paper-facing class name used in configs is `StructureGuidedGatedDecoderHead`.
