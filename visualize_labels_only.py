"""
visualize_labels_only.py
========================
纯标签可视化: 只显示 mask_aligned.nii.gz 中的 nerve/vessel label.

每个 case 一张图:
  - 3 行 (dim0/dim1/dim2 切面)
  - 在有前景的中心 slice 上展示 label
  - 红=nerve(1), 蓝=vessel(2)

用法:
    python visualize_labels_only.py --num_cases 10 --output_dir viz_labels
    python visualize_labels_only.py --case_id 02440781_contra --output_dir viz_labels
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import nibabel as nib
import yaml

from data.dataset import get_train_val_split


def visualize_label(case_id, prepared_dir, output_dir):
    case_dir = os.path.join(prepared_dir, case_id)
    mask_path = os.path.join(case_dir, "mask_aligned.nii.gz")
    if not os.path.exists(mask_path):
        print(f"  跳过: mask 不存在")
        return

    mask = nib.load(mask_path).get_fdata().astype(np.int32)

    nerve_coords = np.argwhere(mask == 1)
    vessel_coords = np.argwhere(mask == 2)
    nerve_vox = len(nerve_coords)
    vessel_vox = len(vessel_coords)
    fg_coords = np.argwhere(mask > 0)

    if len(fg_coords) == 0:
        print(f"  跳过: mask 全为空")
        return

    fg_center = fg_coords.mean(axis=0).astype(int)
    nerve_center = nerve_coords.mean(axis=0).astype(int) if nerve_vox > 0 else None
    vessel_center = vessel_coords.mean(axis=0).astype(int) if vessel_vox > 0 else None

    print(f"  Shape: {mask.shape}, Nerve: {nerve_vox}, Vessel: {vessel_vox}")
    print(f"  FG center: {fg_center.tolist()}")

    label_cmap = ListedColormap(["#000000", "#FF4444", "#4488FF"])

    axis_names = ["Dim0 (Sagittal)", "Dim1 (Coronal)", "Dim2 (Axial)"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(
        f"Case: {case_id}  |  Mask shape: {list(mask.shape)}\n"
        f"Nerve(red): {nerve_vox} vox  |  Vessel(blue): {vessel_vox} vox  |  "
        f"FG center: {fg_center.tolist()}",
        fontsize=13, y=0.98,
    )

    for row, axis in enumerate(range(3)):
        center_idx = fg_center[axis]

        # 3 slices: center-2, center, center+2
        offsets = [-2, 0, 2]
        for col, off in enumerate(offsets):
            idx = np.clip(center_idx + off, 0, mask.shape[axis] - 1)

            if axis == 0:
                sl = mask[idx, :, :]
            elif axis == 1:
                sl = mask[:, idx, :]
            else:
                sl = mask[:, :, idx]

            ax = axes[row, col]
            ax.imshow(sl.T, cmap=label_cmap, origin="lower", vmin=0, vmax=2,
                      interpolation="nearest")
            ax.set_title(f"{axis_names[axis]}, slice={idx}", fontsize=10)

            if col == 0:
                ax.set_ylabel("voxel index")

    legend_items = [
        mpatches.Patch(color="#FF4444", label="Nerve (1)"),
        mpatches.Patch(color="#4488FF", label="Vessel (2)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    out_path = os.path.join(output_dir, f"{case_id}_label.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="viz_labels")
    parser.add_argument("--case_id", default=None)
    parser.add_argument("--num_cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    prepared_dir = cfg["data"]["prepared_dir"]
    os.makedirs(args.output_dir, exist_ok=True)

    if args.case_id:
        case_ids = [args.case_id]
    else:
        train_ids, val_ids = get_train_val_split(
            prepared_dir,
            train_ratio=cfg["data"]["train_val_split"],
            seed=cfg["data"]["random_seed"],
        )
        pool = {"train": train_ids, "val": val_ids, "all": train_ids + val_ids}[args.split]
        rng = np.random.RandomState(args.seed)
        case_ids = list(rng.choice(pool, size=min(args.num_cases, len(pool)), replace=False))

    print(f"可视化 {len(case_ids)} 个 case → {args.output_dir}/\n")

    for case_id in sorted(case_ids):
        print(f"[{case_id}]")
        visualize_label(case_id, prepared_dir, args.output_dir)


if __name__ == "__main__":
    main()
