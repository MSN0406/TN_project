"""
visualize_label_on_wb.py
========================
将 mask label 放回 whole brain 空间, 叠加显示.

每个 case 一张图:
  - 3 行 (sagittal / coronal / axial)
  - 在 mask 前景质心处切 slice
  - 灰度 whole brain + 红色 nerve + 蓝色 vessel 半透明叠加
  - 绿色× = CSV centroid, 红色+ = mask 前景质心

用法:
    python visualize_label_on_wb.py --num_cases 10 --output_dir viz_label_wb
    python visualize_label_on_wb.py --case_id 02440781_contra --output_dir viz_label_wb
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import nibabel as nib
import yaml

from data.dataset import get_train_val_split


def load_case(prepared_dir, case_id):
    case_dir = os.path.join(prepared_dir, case_id)
    with open(os.path.join(case_dir, "info.json")) as f:
        info = json.load(f)

    wb = nib.load(info["nii_path"]).get_fdata().astype(np.float32)
    if wb.ndim == 4:
        wb = wb[:, :, :, 0]

    centroid_ras = np.load(os.path.join(case_dir, "centroid.npy"))
    mask_crop = nib.load(os.path.join(case_dir, "mask_aligned.nii.gz")).get_fdata().astype(np.int32)

    return wb, centroid_ras, mask_crop, info


def place_mask(wb_shape, centroid_ras, mask_crop):
    mask_wb = np.zeros(wb_shape[:3], dtype=np.int32)
    crop_shape = np.array(mask_crop.shape)
    center = np.round(centroid_ras).astype(int)
    starts = center - crop_shape // 2
    ends = starts + crop_shape

    ws = np.maximum(starts, 0)
    we = np.minimum(ends, wb_shape[:3])
    cs = ws - starts
    ce = crop_shape - (ends - we)

    mask_wb[ws[0]:we[0], ws[1]:we[1], ws[2]:we[2]] = \
        mask_crop[cs[0]:ce[0], cs[1]:ce[1], cs[2]:ce[2]]

    fg = np.argwhere(mask_wb > 0)
    fg_center = fg.mean(axis=0) if len(fg) > 0 else center.astype(float)
    return mask_wb, fg_center


def make_overlay(wb_slice, mask_slice):
    """将灰度 wb slice 和 mask slice 合成为 RGBA 图像."""
    vmin, vmax = (np.percentile(wb_slice[wb_slice > 0], [1, 99])
                  if (wb_slice > 0).any() else (0, 1))
    gray = np.clip((wb_slice - vmin) / (vmax - vmin + 1e-8), 0, 1)

    rgb = np.stack([gray, gray, gray], axis=-1)

    # nerve=1 → red overlay
    nerve = mask_slice == 1
    if nerve.any():
        rgb[nerve] = rgb[nerve] * 0.4 + np.array([1.0, 0.2, 0.2]) * 0.6

    # vessel=2 → blue overlay
    vessel = mask_slice == 2
    if vessel.any():
        rgb[vessel] = rgb[vessel] * 0.4 + np.array([0.3, 0.5, 1.0]) * 0.6

    return rgb


def visualize_case(wb, centroid_ras, mask_wb, fg_center, case_id, output_dir):
    center_csv = np.round(centroid_ras).astype(int)
    center_mask = np.round(fg_center).astype(int)

    center_csv = np.clip(center_csv, 0, np.array(wb.shape[:3]) - 1)
    center_mask = np.clip(center_mask, 0, np.array(wb.shape[:3]) - 1)

    axis_names = ["Sagittal (dim0)", "Coronal (dim1)", "Axial (dim2)"]
    # 用 mask fg center 来切 slice, 确保能看到 label
    cut = center_mask

    fig, axes = plt.subplots(3, 1, figsize=(10, 24))
    offset = np.round(fg_center - centroid_ras, 1)
    fig.suptitle(
        f"{case_id}\n"
        f"Green × = CSV centroid {center_csv.tolist()}   "
        f"Red + = Mask centroid {center_mask.tolist()}   "
        f"Offset = {offset.tolist()}",
        fontsize=12, y=0.98,
    )

    for row in range(3):
        idx = cut[row]
        if row == 0:
            wb_sl = wb[idx, :, :]
            mask_sl = mask_wb[idx, :, :]
            csv_xy = (center_csv[1], center_csv[2])
            mask_xy = (center_mask[1], center_mask[2])
        elif row == 1:
            wb_sl = wb[:, idx, :]
            mask_sl = mask_wb[:, idx, :]
            csv_xy = (center_csv[0], center_csv[2])
            mask_xy = (center_mask[0], center_mask[2])
        else:
            wb_sl = wb[:, :, idx]
            mask_sl = mask_wb[:, :, idx]
            csv_xy = (center_csv[0], center_csv[1])
            mask_xy = (center_mask[0], center_mask[1])

        rgb = make_overlay(wb_sl.T, mask_sl.T)

        ax = axes[row]
        ax.imshow(rgb, origin="lower", aspect="equal")
        ax.plot(csv_xy[0], csv_xy[1], "x", color="lime", markersize=14, markeredgewidth=2.5)
        ax.plot(mask_xy[0], mask_xy[1], "+", color="red", markersize=14, markeredgewidth=2.5)
        ax.set_title(f"{axis_names[row]}, slice={idx}", fontsize=12)

    legend_items = [
        mpatches.Patch(color="lime", label="CSV centroid"),
        mpatches.Patch(color="red", label="Mask fg centroid"),
        mpatches.Patch(color="#FF4444", label="Nerve (1)"),
        mpatches.Patch(color="#4488FF", label="Vessel (2)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(output_dir, f"{case_id}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="viz_label_wb")
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
        try:
            wb, centroid_ras, mask_crop, info = load_case(prepared_dir, case_id)
        except Exception as e:
            print(f"  跳过: {e}")
            continue

        mask_wb, fg_center = place_mask(wb.shape, centroid_ras, mask_crop)
        print(f"  CSV centroid:  {centroid_ras.astype(int).tolist()}")
        print(f"  Mask centroid: {np.round(fg_center, 1).tolist()}")
        print(f"  Offset:        {np.round(fg_center - centroid_ras, 1).tolist()}")

        visualize_case(wb, centroid_ras, mask_wb, fg_center, case_id, args.output_dir)


if __name__ == "__main__":
    main()
