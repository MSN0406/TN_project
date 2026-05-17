"""
verify_alignment.py
===================
直接验证 centroid crop 和 mask 是否对齐.

对每个 case:
  1. 从 whole brain 以 centroid 为中心裁出 64³ ROI (与训练时完全一致)
  2. 加载 mask_aligned.nii.gz (64³)
  3. 在 3 个正交面 (通过前景质心) 叠加显示:
     - 列 1: Brain crop (灰度)
     - 列 2: Mask (nerve=红, vessel=蓝)
     - 列 3: 叠加 (brain + mask overlay)
  如果对齐正确, 列 3 中 nerve/vessel 轮廓应精确贴合 brain 上的结构.

用法:
    python verify_alignment.py --num_cases 10 --output_dir viz_alignment_crop
    python verify_alignment.py --case_id 02440781_contra
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


def hard_crop(volume, center, crop_size=64):
    """从 volume 中以 center 为中心裁出 crop_size³, 与训练时 hard_crop_3d 一致."""
    half = crop_size // 2
    shape = np.array(volume.shape[:3])
    center = np.round(center).astype(int)

    starts = np.clip(center - half, 0, shape - crop_size)
    ends = starts + crop_size
    return volume[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]


def normalize_gray(vol_slice):
    """将 brain slice 归一化到 [0, 1] 灰度."""
    fg = vol_slice > 0
    if fg.sum() > 10:
        vmin, vmax = np.percentile(vol_slice[fg], [1, 99])
    else:
        vmin, vmax = vol_slice.min(), vol_slice.max()
    return np.clip((vol_slice - vmin) / (vmax - vmin + 1e-8), 0, 1)


def make_mask_rgb(mask_slice):
    """Mask slice → RGB: nerve(1)=红, vessel(2)=蓝, bg=黑."""
    h, w = mask_slice.shape
    rgb = np.zeros((h, w, 3))
    rgb[mask_slice == 1] = [1.0, 0.2, 0.2]
    rgb[mask_slice == 2] = [0.3, 0.5, 1.0]
    return rgb


def make_overlay(brain_slice, mask_slice, alpha=0.55):
    """Brain 灰度 + mask 半透明叠加."""
    gray = normalize_gray(brain_slice)
    rgb = np.stack([gray, gray, gray], axis=-1)

    nerve = mask_slice == 1
    vessel = mask_slice == 2

    if nerve.any():
        rgb[nerve] = rgb[nerve] * (1 - alpha) + np.array([1.0, 0.2, 0.2]) * alpha
    if vessel.any():
        rgb[vessel] = rgb[vessel] * (1 - alpha) + np.array([0.3, 0.5, 1.0]) * alpha

    # 轮廓线: 在 mask 边界画白边, 更容易看到对齐偏差
    from scipy.ndimage import binary_dilation
    for label_val in [1, 2]:
        region = mask_slice == label_val
        if not region.any():
            continue
        dilated = binary_dilation(region, iterations=1)
        contour = dilated & ~region
        color = [1.0, 0.5, 0.5] if label_val == 1 else [0.5, 0.7, 1.0]
        rgb[contour] = color

    return rgb


def visualize_case(brain_crop, mask_crop, case_id, output_dir, info=None):
    """生成一个 case 的对齐验证图."""

    # 找前景质心来决定切片位置
    fg_coords = np.argwhere(mask_crop > 0)
    if len(fg_coords) == 0:
        print(f"  [WARN] {case_id}: mask 没有前景, 跳过")
        return
    fg_center = fg_coords.mean(axis=0)
    cut = np.round(fg_center).astype(int)
    cut = np.clip(cut, 0, np.array(mask_crop.shape) - 1)

    axis_names = ["Sagittal (dim0)", "Coronal (dim1)", "Axial (dim2)"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    offset_from_center = fg_center - np.array(mask_crop.shape) / 2
    fig.suptitle(
        f"{case_id} — ROI crop vs Mask overlay verification\n"
        f"Foreground center in 64³: [{fg_center[0]:.1f}, {fg_center[1]:.1f}, {fg_center[2]:.1f}]   "
        f"Offset from center: [{offset_from_center[0]:.1f}, {offset_from_center[1]:.1f}, {offset_from_center[2]:.1f}]",
        fontsize=12, y=0.98,
    )

    for row in range(3):
        idx = cut[row]
        if row == 0:
            b_sl = brain_crop[idx, :, :]
            m_sl = mask_crop[idx, :, :]
        elif row == 1:
            b_sl = brain_crop[:, idx, :]
            m_sl = mask_crop[:, idx, :]
        else:
            b_sl = brain_crop[:, :, idx]
            m_sl = mask_crop[:, :, idx]

        # 列 1: Brain crop
        gray = normalize_gray(b_sl.T)
        axes[row, 0].imshow(np.stack([gray]*3, -1), origin="lower", aspect="equal")
        axes[row, 0].set_title(f"Brain crop — {axis_names[row]} [{idx}]", fontsize=10)

        # 列 2: Mask
        mask_rgb = make_mask_rgb(m_sl.T)
        axes[row, 1].imshow(mask_rgb, origin="lower", aspect="equal")
        axes[row, 1].set_title(f"Mask — {axis_names[row]} [{idx}]", fontsize=10)

        # 列 3: Overlay
        overlay = make_overlay(b_sl.T, m_sl.T)
        axes[row, 2].imshow(overlay, origin="lower", aspect="equal")
        axes[row, 2].set_title(f"Overlay — {axis_names[row]} [{idx}]", fontsize=10)

    for ax in axes.flat:
        ax.axis("off")

    legend_items = [
        mpatches.Patch(color="#FF3333", label="Nerve (1)"),
        mpatches.Patch(color="#4D80FF", label="Vessel (2)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(output_dir, f"{case_id}_align.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="viz_alignment_crop")
    parser.add_argument("--case_id", default=None)
    parser.add_argument("--num_cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="val", choices=["train", "val", "all"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    prepared_dir = cfg["data"]["prepared_dir"]
    crop_size = cfg["segmentation"]["crop_size"]
    if isinstance(crop_size, list):
        crop_size = crop_size[0]

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

    print(f"对齐验证: {len(case_ids)} cases → {args.output_dir}/")
    print(f"Crop size: {crop_size}\n")

    for case_id in sorted(case_ids):
        print(f"[{case_id}]")
        case_dir = os.path.join(prepared_dir, case_id)
        try:
            with open(os.path.join(case_dir, "info.json")) as f:
                info = json.load(f)

            wb = nib.load(info["nii_path"]).get_fdata().astype(np.float32)
            if wb.ndim == 4:
                wb = wb[:, :, :, 0]

            centroid_ras = np.load(os.path.join(case_dir, "centroid.npy"))
            mask = nib.load(os.path.join(case_dir, "mask_aligned.nii.gz")).get_fdata().astype(np.int32)

        except Exception as e:
            print(f"  跳过: {e}")
            continue

        brain_crop = hard_crop(wb, centroid_ras, crop_size)

        print(f"  Centroid:    {centroid_ras.astype(int).tolist()}")
        print(f"  Brain shape: {wb.shape[:3]}")
        print(f"  Crop shape:  {brain_crop.shape}")
        print(f"  Mask shape:  {mask.shape}")

        if brain_crop.shape != mask.shape:
            print(f"  [WARN] shape 不匹配! crop={brain_crop.shape} vs mask={mask.shape}")
            continue

        visualize_case(brain_crop, mask, case_id, args.output_dir, info)

    print(f"\n完成! 查看 {args.output_dir}/ 目录")
    print("如果叠加图 (右列) 中 mask 轮廓精确贴合 brain 结构 → 对齐正确")
    print("如果有系统性偏移 → 坐标变换有 bug, 需要修复")


if __name__ == "__main__":
    main()
