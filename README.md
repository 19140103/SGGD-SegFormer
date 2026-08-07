# SGGD-SegFormer

Code release for **Structure-Guided Gated Decoding for Lightweight Bridge Crack Segmentation**.

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

## What is not included

The repository intentionally does not include unpublished manuscript files, paper PDFs, MDPI templates, paper figures, raw datasets, trained checkpoints, logs, or result folders.

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

Raw datasets are not included. Prepare data with:

```bash
python scripts/prepare_segcodebrim_mmseg.py --raw-root /path/to/SegCODEBRIM --out-root data/SegCODEBRIM_mmseg
python scripts/prepare_dacl10k_crack_mmseg.py --source-root /path/to/dacl10k --output-root data/dacl10k_crack_mmseg
```

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

## Evaluation

Run one-pass evaluation with ODS threshold sweeping, fixed-threshold metrics, complexity, and FPS:

```bash
python scripts/evaluate_crack_model_once.py configs/bridge_crack/sggd_segformer_b0_segcodebrim_512x512_binary_bce_tversky.py /path/to/checkpoint.pth --data-root data/SegCODEBRIM_mmseg --split test
```

Run threshold sweeping only:

```bash
python scripts/eval_threshold_sweep.py configs/bridge_crack/sggd_segformer_b0_segcodebrim_512x512_binary_bce_tversky.py /path/to/checkpoint.pth --data-root data/SegCODEBRIM_mmseg --split test
```

## Notes

The decoder source file keeps the historical filename `spg_segformer_head.py` for checkpoint compatibility. The paper-facing class name used in configs is `StructureGuidedGatedDecoderHead`.
