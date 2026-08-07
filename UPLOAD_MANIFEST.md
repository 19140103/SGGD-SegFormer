# Upload Manifest

This folder is prepared as a code-only GitHub upload candidate for SGGD-SegFormer.

## Include

- `mmseg/`
  - MMSegmentation source tree with SGGD-related decode head, binary losses, and hook additions.
- `configs/_base_/`
  - MMSegmentation base config files required by the bridge crack configs.
- `configs/bridge_crack/`
  - Curated experiment configs for SegCODEBRIM, DACL10K low-annotation adaptation, ablation settings, and comparison baselines.
- `tools/`
  - MMSegmentation train/test entry points.
- `scripts/`
  - Dataset preparation and metric evaluation utilities needed for reproduction.
- `requirements/`, `requirements.txt`, `setup.py`, `setup.cfg`, `MANIFEST.in`
  - Environment and packaging files.
- `LICENSE`, `.gitignore`, `README.md`
  - Repository metadata.

## Exclude

- Paper source files such as `main.tex`.
- Manuscript PDF files.
- MDPI template files.
- Paper figures and subfigures.
- Raw datasets.
- Checkpoints, weights, logs, and result folders.
- Temporary caches such as `__pycache__` and `.pyc` files.

## Files Curated From The Original Working Tree

- Removed temporary/sanity configs from the upload candidate.
- Converted local absolute paths such as `G:/Bridge/data/...` and `G:/Bridge/results/...` to relative `data/...` and `work_dirs/...`.
- Replaced local low-annotation fine-tuning checkpoints with `load_from = None`; checkpoint paths should be provided by the user at runtime.
- Renamed public ablation config filenames from historical `spg_...` names to `sggd_...` names.
