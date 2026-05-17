"""
visualize_centroid_vs_mask.py
=============================
在 whole brain 上同时可视化 CSV centroid 位置和 mask label 位置,
检查两者是否空间对齐.

对每个 case 生成一张图:
  - 3 行 (sagittal/coronal/axial 切面)
  - 每行 2 列: 左=在 centroid 处切 whole brain + 标注 centroid 十字;
               右=同一切面叠加 mask (放回 whole brain 空间)
  - 绿色十字 = CSV centroid
  - 红色十字 = mask 前景质心 (如果偏移说明数据未对齐)

用法:
    conda activate tnproject
    python visualize_centroid_vs_mask.py --num_cases 10 --output_dir viz_alignment
    python visualize_centroid_vs_mask.py --case_id 02440781_contra --output_dir viz_alignment
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


def load_case_data(prepared_dir, case_id):
    """加载单个 case 的 whole brain, centroid, mask."""
    case_dir = os.path.join(prepared_dir, case_id)

    with open(os.path.join(case_dir, "info.json")) as f:
        info = json.load(f)

    nii_path = info["nii_path"]
    nii = nib.load(nii_path)
    wb = nii.get_fdata().astype(np.float32)
    if wb.ndim == 4:
        wb = wb[:, :, :, 0]

    centroid_ras = np.load(os.path.join(case_dir, "centroid.npy"))

    mask_path = os.path.join(case_dir, "mask_aligned.nii.gz")
    mask_crop = nib.load(mask_path).get_fdata().astype(np.int32)

    return wb, centroid_ras, mask_crop, info


def place_mask_in_wb(wb_shape, centroid_ras, mask_crop):
    """
    将 mask_crop (64³) 放回 whole brain 空间, 以 centroid_ras 为中心.

    返回:
        mask_wb: 与 whole brain 同尺寸的 mask volume
        mask_centroid_wb: mask 前景的质心 (在 whole brain 空间中)
    """
    mask_wb = np.zeros(wb_shape[:3], dtype=np.int32)
    crop_shape = np.array(mask_crop.shape)
    center = np.round(centroid_ras).astype(int)

    starts = center - crop_shape // 2
    ends = starts + crop_shape

    # 处理边界 clipping
    wb_starts = np.maximum(starts, 0)
    wb_ends = np.minimum(ends, wb_shape[:3])
    crop_starts = wb_starts - starts
    crop_ends = crop_shape - (ends - wb_ends)

    mask_wb[
        wb_starts[0]:wb_ends[0],
        wb_starts[1]:wb_ends[1],
        wb_starts[2]:wb_ends[2],
    ] = mask_crop[
        crop_starts[0]:crop_ends[0],
        crop_starts[1]:crop_ends[1],
        crop_starts[2]:crop_ends[2],
    ]

    # mask 前景质心 (在 whole brain 空间)
    fg_coords = np.argwhere(mask_wb > 0)
    if len(fg_coords) > 0:
        mask_centroid_wb = fg_coords.mean(axis=0)
    else:
        mask_centroid_wb = center.astype(float)

    # 分类质心
    nerve_coords = np.argwhere(mask_wb == 1)
    vessel_coords = np.argwhere(mask_wb == 2)
    nerve_centroid = nerve_coords.mean(axis=0) if len(nerve_coords) > 0 else None
    vessel_centroid = vessel_coords.mean(axis=0) if len(vessel_coords) > 0 else None

    return mask_wb, mask_centroid_wb, nerve_centroid, vessel_centroid


def visualize_case(wb, centroid_ras, mask_wb, mask_centroid_wb,
                   nerve_centroid, vessel_centroid, case_id, output_path):
    """
    生成一张对比图: whole brain 切面 + centroid 标记 + mask overlay.
    """
    center = np.round(centroid_ras).astype(int)
    center = np.clip(center, 0, np.array(wb.shape[:3]) - 1)

    axis_names = ["Sagittal (dim0)", "Coronal (dim1)", "Axial (dim2)"]
    # 每个轴: 在 centroid 位置切一个 slice
    slices_wb = [
        wb[center[0], :, :],
        wb[:, center[1], :],
        wb[:, :, center[2]],
    ]
    slices_mask = [
        mask_wb[center[0], :, :],
        mask_wb[:, center[1], :],
        mask_wb[:, :, center[2]],
    ]
    # centroid 在每个切面上的 2D 坐标
    centroid_2d = [
        (center[1], center[2]),  # sagittal: (cor, axi)
        (center[0], center[2]),  # coronal: (sag, axi)
        (center[0], center[1]),  # axial: (sag, cor)
    ]
    # mask centroid 在每个切面上的 2D 坐标
    mask_c = np.round(mask_centroid_wb).astype(int)
    mask_centroid_2d = [
        (mask_c[1], mask_c[2]),
        (mask_c[0], mask_c[2]),
        (mask_c[0], mask_c[1]),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle(f"Case: {case_id}\n"
                 f"CSV centroid (green): {centroid_ras.astype(int).tolist()}\n"
                 f"Mask fg centroid (red): {np.round(mask_centroid_wb, 1).tolist()}\n"
                 f"Offset: {np.round(mask_centroid_wb - centroid_ras, 1).tolist()}",
                 fontsize=13, y=0.98)

    # 颜色映射: 0=transparent, 1=nerve(red), 2=vessel(blue)
    from matplotlib.colors import ListedColormap
    mask_cmap = ListedColormap(["none", "#FF4444", "#4488FF"])

    for row in range(3):
        img = slices_wb[row].T
        mask_img = slices_mask[row].T
        cy, cx = centroid_2d[row]
        my, mx = mask_centroid_2d[row]

        # 左列: whole brain + centroid 十字线
        ax = axes[row, 0]
        vmin, vmax = np.percentile(img[img > 0], [1, 99]) if (img > 0).any() else (0, 1)
        ax.imshow(img, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.axhline(cx, color="lime", linewidth=0.8, alpha=0.7)
        ax.axvline(cy, color="lime", linewidth=0.8, alpha=0.7)
        ax.plot(cy, cx, "x", color="lime", markersize=12, markeredgewidth=2, label="CSV centroid")
        ax.set_title(f"{axis_names[row]} — WB + Centroid", fontsize=11)
        ax.set_ylabel("slice index")
        ax.legend(loc="upper right", fontsize=8)

        # 右列: whole brain + mask overlay + 双 centroid
        ax = axes[row, 1]
        ax.imshow(img, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        masked = np.ma.masked_where(mask_img == 0, mask_img)
        ax.imshow(masked, cmap=mask_cmap, origin="lower", alpha=0.6,
                  vmin=0, vmax=2)
        ax.plot(cy, cx, "x", color="lime", markersize=12, markeredgewidth=2)
        ax.plot(my, mx, "+", color="red", markersize=14, markeredgewidth=2)

        # 分类质心
        if nerve_centroid is not None:
            nc = np.round(nerve_centroid).astype(int)
            nc_2d = [(nc[1], nc[2]), (nc[0], nc[2]), (nc[0], nc[1])][row]
            ax.plot(nc_2d[0], nc_2d[1], "o", color="#FF4444", markersize=6,
                    markeredgewidth=1.5, fillstyle="none")
        if vessel_centroid is not None:
            vc = np.round(vessel_centroid).astype(int)
            vc_2d = [(vc[1], vc[2]), (vc[0], vc[2]), (vc[0], vc[1])][row]
            ax.plot(vc_2d[0], vc_2d[1], "s", color="#4488FF", markersize=6,
                    markeredgewidth=1.5, fillstyle="none")

        legend_items = [
            mpatches.Patch(color="lime", label="CSV centroid"),
            mpatches.Patch(color="red", label="Mask fg centroid"),
            mpatches.Patch(color="#FF4444", label="Nerve (class 1)"),
            mpatches.Patch(color="#4488FF", label="Vessel (class 2)"),
        ]
        ax.legend(handles=legend_items, loc="upper right", fontsize=7)
        ax.set_title(f"{axis_names[row]} — WB + Mask overlay", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="可视化 CSV centroid 与 mask 在 whole brain 中的位置")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output_dir", default="viz_alignment")
    parser.add_argument("--case_id", default=None, help="指定单个 case (如 02440781_contra)")
    parser.add_argument("--num_cases", type=int, default=10, help="随机选取的 case 数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="val", choices=["train", "val", "all"],
                        help="从哪个 split 选 case")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    prepared_dir = cfg["data"]["prepared_dir"]
    os.makedirs(args.output_dir, exist_ok=True)

    # 选择 case
    if args.case_id:
        case_ids = [args.case_id]
    else:
        train_ids, val_ids = get_train_val_split(
            prepared_dir,
            train_ratio=cfg["data"]["train_val_split"],
            seed=cfg["data"]["random_seed"],
        )
        if args.split == "train":
            pool = train_ids
        elif args.split == "val":
            pool = val_ids
        else:
            pool = train_ids + val_ids

        rng = np.random.RandomState(args.seed)
        case_ids = list(rng.choice(pool, size=min(args.num_cases, len(pool)), replace=False))

    print(f"将可视化 {len(case_ids)} 个 case, 输出到 {args.output_dir}/")
    print()

    offset_stats = []

    for case_id in sorted(case_ids):
        print(f"[{case_id}]")
        try:
            wb, centroid_ras, mask_crop, info = load_case_data(prepared_dir, case_id)
        except Exception as e:
            print(f"  跳过: {e}")
            continue

        mask_wb, mask_centroid_wb, nerve_c, vessel_c = place_mask_in_wb(
            wb.shape, centroid_ras, mask_crop)

        offset = mask_centroid_wb - centroid_ras
        offset_stats.append({"case": case_id, "offset": offset})
        print(f"  CSV centroid:  {centroid_ras.astype(int).tolist()}")
        print(f"  Mask centroid: {np.round(mask_centroid_wb, 1).tolist()}")
        print(f"  Offset:        {np.round(offset, 1).tolist()}")
        print(f"  |Offset|:      {np.linalg.norm(offset):.1f} voxels")

        nerve_vox = int((mask_crop == 1).sum())
        vessel_vox = int((mask_crop == 2).sum())
        print(f"  Nerve voxels:  {nerve_vox}, Vessel voxels: {vessel_vox}")

        output_path = os.path.join(args.output_dir, f"{case_id}.png")
        visualize_case(wb, centroid_ras, mask_wb, mask_centroid_wb,
                       nerve_c, vessel_c, case_id, output_path)

    # 汇总统计
    if len(offset_stats) > 1:
        offsets = np.array([s["offset"] for s in offset_stats])
        dists = np.linalg.norm(offsets, axis=1)
        print(f"\n{'='*50}")
        print(f"偏移统计 ({len(offset_stats)} cases):")
        print(f"  Mean |offset|:  {dists.mean():.2f} voxels")
        print(f"  Max  |offset|:  {dists.max():.2f} voxels")
        print(f"  Median:         {np.median(dists):.2f} voxels")
        print(f"  Per-axis mean:  {np.mean(np.abs(offsets), axis=0).round(2).tolist()}")
        print(f"  Per-axis std:   {np.std(offsets, axis=0).round(2).tolist()}")


if __name__ == "__main__":
    main()
