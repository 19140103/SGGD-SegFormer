# Model zoo

The [v1.0.0 release](https://github.com/19140103/SGGD-SegFormer/releases/tag/v1.0.0)
contains 20 selected checkpoints supporting the main controlled comparison and
the paired DACL10K adaptation experiments in DOI
[10.1038/s41598-026-80270-1](https://doi.org/10.1038/s41598-026-80270-1).

## Main SegCODEBRIM checkpoints

| Model | ODS F1 (%) | ODS IoU (%) | Selected iteration | Download |
| --- | --- | --- | --- | --- |
| SGGD-SegFormer | 48.54 | 32.05 | 12000 | [Weights](https://github.com/19140103/SGGD-SegFormer/releases/download/v1.0.0/sggd_segformer_b0_segcodebrim_best.pth) |
| SegFormer-B0 | 47.02 | 30.74 | 11500 | [Weights](https://github.com/19140103/SGGD-SegFormer/releases/download/v1.0.0/segformer_b0_segcodebrim_best.pth) |

These are the checkpoints selected by validation mIoU, not by test-set scores.
The same source checkpoints are used for zero-shot evaluation and initialization
of all target-domain repetitions. ODS metrics use the manuscript's threshold
sweep; ODS IoU uses the threshold selected for ODS F1. A test-set ODS threshold
is a reporting operating point, not an independently calibrated deployment
threshold. Fixed-threshold metrics use 0.5.

Example evaluation after downloading the SGGD weights to `checkpoints/`:

```bash
python scripts/evaluate_crack_model_resized.py \
  configs/bridge_crack/sggd_segformer_b0_segcodebrim_512x512_binary_bce_tversky.py \
  checkpoints/sggd_segformer_b0_segcodebrim_best.pth \
  --model-name SGGD-SegFormer \
  --data-root data/SegCODEBRIM_mmseg --split test --img-suffix .png --eval-size 512
```

The baseline uses the matching `segformer_b0_segcodebrim_512x512_binary_bce_tversky.py`
config. Historical `spg` names in decoder source and original checkpoints are
retained internally for compatibility; public configs use
`StructureGuidedGatedDecoderHead`.

## DACL10K checkpoints

All 18 best-validation-mIoU checkpoints are included, without selecting a
preferred repetition. Filenames are
`{sggd_segformer_b0|segformer_b0}_dacl10k_seed{01|02|03}_pct{05|10|20}.pth`.
Each checkpoint's direct download URL, config, selected iteration, split seed,
training seed, metrics and SHA-256 checksum are listed in
[checkpoint_manifest.json](checkpoint_manifest.json).

| Repetition | Split seed | Fine-tuning seed |
| --- | --- | --- |
| seed01 | 20260923 | 31001 |
| seed02 | 20260924 | 31002 |
| seed03 | 20260925 | 31003 |

The processed binary subset contains 975 images. The fixed held-out test set
contains 587 images, and 388 images form the sampling pool. Training/validation
counts are 50/50, 97/97 and 194/194 for the 5%, 10% and 20% training annotation
ratios. Validation labels are additional to those training ratios. Both models
use identical splits within each paired repetition; each has a 3000-iteration
fine-tuning budget and checkpoints selected using validation mIoU.

Use the matching DACL10K config recorded in the manifest and evaluate on the
fixed test root, not on a per-repetition training/validation root. For example:

```bash
python scripts/evaluate_crack_model_resized.py \
  configs/bridge_crack/sggd_segformer_b0_dacl10k_fewshot10_512x512_bce_tversky.py \
  checkpoints/sggd_segformer_b0_dacl10k_seed01_pct10.pth \
  --model-name SGGD-seed01-pct10 \
  --data-root data/dacl10k_crack_fewshot20_mmseg --split test \
  --img-suffix .jpg --eval-size 512 --skip-structure --skip-complexity
```

[Per-checkpoint metrics](results/checkpoint_metrics.csv) and
[DACL10K means and sample standard deviations](results/dacl10k_mean_sd.csv)
are the original evaluation records underlying the paper, not new evaluation
results. At 5% and 10% SGGD has higher mean ODS scores; at 20% the baseline has
higher mean ODS scores. The release does not change these results.

## Splits and data access

The `splits/` directory records sample identifiers without distributing dataset
images or masks. SegCODEBRIM identifiers are listed for train/validation/test;
DACL10K identifiers include the shared test set and all nested train/validation
splits. Entries are image stems, without extensions. Preparation and split
generation commands remain in the main README.

- SegCODEBRIM: https://zenodo.org/records/10071534
- DACL10K toolkit: https://github.com/phiyodr/dacl10k-toolkit
- DACL10K source archive used for the binary subset:
  https://dacl10k.s3.eu-central-1.amazonaws.com/dacl10k-challenge/dacl10k_v2_devphase.zip

Obtain third-party datasets from their original sources under their applicable
terms. This repository does not redistribute those datasets.

## Format and verification

The release weights are PyTorch dictionaries with `state_dict` and minimal
public `meta` fields. All tensor names, shapes, dtypes and values are preserved
exactly from the original selected checkpoints. Training paths, logs, optimizer
state and scheduler state are excluded. These are inference/fine-tuning weights,
not exact optimizer-resume snapshots.

All 20 exported checkpoints were checked for tensor equality against their
originals, strict loading with the public model implementation, and a finite
512x512 forward pass. This is a packaging/compatibility check, not a rerun of the
complete experimental evaluation.

The verified environment is Python 3.10, PyTorch 2.1.2+cu118, MMCV 2.1.0 and
MMEngine 0.10.7. The repository's current MMSegmentation code requires MMCV below
2.2.0. Match the MMCV build to the PyTorch/CUDA environment.

Compare downloaded files against [SHA256SUMS.txt](SHA256SUMS.txt), also attached
to the release. Only load trusted PyTorch files; these release files support
`torch.load(path, map_location='cpu', weights_only=True)`.
