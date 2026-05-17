# TN-LocSeg: Localization-Guided Trigeminal Nerve & Vessel Segmentation in 3D MRI

End-to-end joint localization + multi-class segmentation framework for the
trigeminal nerve (TN) and surrounding vessels in 3D MRI, built on top of
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet).

Companion code for the NeurIPS 2026 manuscript
*Localization-Guided Trigeminal Nerve and Vessel Segmentation in 3D MRI with
Cross-Institution Generalization Analysis* (see
[`Formatting_Instructions_For_NeurIPS_2026/tn_locseg_paper.tex`](Formatting_Instructions_For_NeurIPS_2026/tn_locseg_paper.tex)).

---

## What this repo provides

- A custom `nnUNetTrainerLoc` that wraps nnU-Net's segmentation backbone with
  a bilateral 3D localization head and a differentiable ROI-cropping operator,
  trained jointly end-to-end ([`nnunet_loc_trainer.py`](nnunet_loc_trainer.py)).
- Network modules in [`models/`](models/) (LocEncoder3D, LocSegWrapper,
  differentiable cropping).
- Run scripts for every training / fine-tuning / inference experiment in the
  paper, under [`scripts/`](scripts/).
- Evaluation, verification, and visualization helpers at the repo root
  (`verify_*.py`, `visualize_*.py`, `viz_*.py`, `eval_*.py`, `export_*.py`,
  `scripts/scan_*.py`).
- The NeurIPS 2026 paper source + all figures.

The private CISS dataset and the OpenNeuro derivative are **not** included
(IRB / privacy). All training/inference paths are env-var driven so you can
point the code at your own data layout.

---

## Setup

```bash
# 1. Create env
conda create -n tnproject python=3.10 -y
conda activate tnproject

# 2. Install dependencies
pip install -r requirements.txt
# Also install nnUNetv2 (https://github.com/MIC-DKFZ/nnUNet)
pip install nnunetv2
```

### Environment variables

All scripts honor these (with sensible local defaults inside the repo):

| Variable | Purpose | Default |
|---|---|---|
| `TN_ROOT` | Repo root (auto-detected from script location) | repo dir |
| `TN_RAW_DATA_DIR` | Root of private raw MRI dataset (CISS) | — *(must set)* |
| `TN_PREPARED_DATA_DIR` | Pre-processed per-case directories | `${TN_ROOT}/prepared_data` |
| `nnUNet_raw` | nnUNet raw dataset root | `${TN_ROOT}/nnUNet_data/nnUNet_raw` |
| `nnUNet_preprocessed` | nnUNet preprocessed root | `${TN_ROOT}/nnUNet_data/nnUNet_preprocessed` |
| `nnUNet_results` | nnUNet results root | `${TN_ROOT}/nnUNet_data/nnUNet_results` |
| `CUDA_VISIBLE_DEVICES` | GPU ids | `0` |
| `PYTHON` | Python interpreter for shell wrappers | `python` |

Many task-specific knobs (e.g. `TN_WB_TARGET_SPACING`, `TN_PHASE3_VAL_USE_GT_CROP`,
`TN_PHASE3_VESSEL_OVERSAMPLE_PROB`, `TN_LOC_PHASE3_AUG`) are set inside the
shell scripts under [`scripts/`](scripts/) — read those scripts to see the
exact configuration used for each paper experiment.

---

## Expected on-disk data layout

Customize via the env vars above. The pipeline expects:

```
${TN_RAW_DATA_DIR}/                        # set by you
  NIFTI/                                   # raw volumes (.nii.gz)
  2020-22_MRIs_Cropped_and_Segmentations/
    202022_centroids_ipsilateral.csv
    202022_centroids_contralateral.csv

${TN_PREPARED_DATA_DIR}/                   # default: ./prepared_data
  <case_id>/
    image.nii.gz
    label.nii.gz
    centroid.npy
    info.json

${nnUNet_raw}/Dataset001_TN/               # default: ./nnUNet_data/nnUNet_raw/Dataset001_TN/
  imagesTr/  imagesTs/  labelsTr/  dataset.json  case_mapping.json

${nnUNet_preprocessed}/Dataset001_TN/      # produced by `nnUNetv2_plan_and_preprocess`
  nnUNetPlans.json  splits_final.json  gt_segmentations/  ...

${nnUNet_results}/Dataset001_TN/...        # where ckpts land
```

---

## Reproducing the paper results

### 1. Preprocess (one-time)

```bash
# Convert raw → nnUNet raw format
python convert_to_nnunet.py

# Run nnUNet planning/preprocessing
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity

# Prepare per-case directories with centroids
python data/prepare_dataset.py            # in-house CISS
python data/prepare_openneuro_tn.py       # OpenNeuro T1w/T2w
```

### 2. Main paper training runs

| Paper row | Script |
|---|---|
| LGMS, no loc aug (in-house) | `scripts/run_p3_scratch_mix0604_v3.sh` |
| LGMS, **w/ loc aug** (in-house, main) | `scripts/run_p3_scratch_mix0604_locaug.sh` |
| OpenNeuro fine-tune from no-aug ckpt | `scripts/run_finetune_openneuro_t2w.sh` |
| OpenNeuro fine-tune from loc-aug ckpt | `scripts/run_finetune_openneuro_t2w_from_locaug.sh` |
| OpenNeuro zero-shot | `python scripts/infer_openneuro_zeroshot.py --ckpt <path>` |

### 3. Generate paper figures from a trained checkpoint

```bash
# Per-case dice/loc-err CSV + sample axial slices
python scripts/scan_val_for_figures.py \
  --ckpt <phase3_best.pth> \
  --out_dir Formatting_Instructions_For_NeurIPS_2026/regen/ \
  --gpu 0

# Render fig_loc_verify.png + fig_seg_dice.png
python scripts/plot_fig_loc_verify.py \
  --in_dir Formatting_Instructions_For_NeurIPS_2026/regen/ \
  --out_png Formatting_Instructions_For_NeurIPS_2026/fig_loc_verify.png

python scripts/plot_fig_seg_dice.py \
  --in_csv Formatting_Instructions_For_NeurIPS_2026/regen/val_dice_per_case.csv \
  --out_png Formatting_Instructions_For_NeurIPS_2026/fig_seg_dice.png
```

### 4. Build the paper PDF

```bash
cd Formatting_Instructions_For_NeurIPS_2026/
pdflatex tn_locseg_paper.tex
bibtex   tn_locseg_paper
pdflatex tn_locseg_paper.tex
pdflatex tn_locseg_paper.tex
```

---

## Repo layout

```
nnunet_loc_trainer.py        Custom nnU-Net trainer (joint loc + seg head)
train_nnunet_loc.py          Main training entry point
train.py                     Alternative trainer entry
models/                      Network modules (LocEncoder3D, etc.)
training/                    Loss functions + trainer helpers
data/                        Dataset wrappers + preprocessing scripts
configs/                     YAML configs (paths use ${...} placeholders — fill in)
config.yaml                  Default config
scripts/
  run_*.sh                   Training run wrappers
  continue_*.sh              Resume from a checkpoint
  finetune_openneuro.py      OpenNeuro fine-tuning entry
  infer_*.py                 Inference scripts
  scan_*.py                  Bulk per-case eval scripts
  plot_fig_*.py              Paper figure renderers
tools/                       Test helpers + nnUNet progress replotter
verify_*.py                  Data/alignment verification
visualize_*.py viz_*.py      Visualization (axial overlays, dashboards, etc.)
eval_*.py export_*.py        Metric export / dashboards
analyze_*.py                 Dataset analyses
diag_*.py                    Diagnostic / debugging utilities
fix_*.py                     Data-fix scripts
Formatting_Instructions_For_NeurIPS_2026/
  tn_locseg_paper.tex        Paper source
  fig_*.png                  Paper figures
  Network Ar.png             Architecture diagram
  neurips_2026.sty           NeurIPS style file
```

---

## Notes on data & ethics

- The in-house 3T CISS cohort is IRB-protected and **not released**.
- The OpenNeuro TN dataset is publicly available; we provide preprocessing
  scripts only.
- This is research code intended as a planning aid; it is **not** clinically
  validated. See the ethics statement in the paper for details.

## Citation

If you use this code, please cite the NeurIPS 2026 paper (BibTeX entry will be
added on publication).

## License

Code: MIT (suggested — pick your own). Paper text and figures: CC-BY-4.0.
