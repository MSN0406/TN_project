"""
fix_data_alignment.py
=====================
修复 centroid 与 mask 之间的空间对齐问题.

问题: CSV centroid 和 mask TIFF 来自不同来源, 导致:
  - ROI 裁剪 (以 centroid 为中心) 的内容
  - mask 标注的内容
  两者之间存在系统性偏移 (2-6 voxels)

修复策略:
  1. 计算 mask 前景 (nerve+vessel) 在 64³ 空间中的质心
  2. 调整 centroid_ras, 使 ROI 裁剪以前景为中心
  3. 平移 mask 内容, 使前景居中于 64³ 体积
  4. 两者都对齐到前景质心 → 消除偏移

备份: 原始文件保存为 centroid_orig.npy, mask_aligned_orig.nii.gz

用法:
    python fix_data_alignment.py --config configs/default.yaml
    python fix_data_alignment.py --config configs/default.yaml --dry_run  # 仅统计不修改
"""

import argparse
import json
import os
import shutil

import nibabel as nib
import numpy as np
import yaml
from scipy.ndimage import shift as ndshift


def compute_fg_centroid(mask, classes=(1, 2)):
    """计算指定类别的前景质心."""
    fg = np.isin(mask, classes)
    coords = np.argwhere(fg)
    if len(coords) == 0:
        return None
    return coords.mean(axis=0)


def shift_mask(mask, shift_vec):
    """
    整数平移 mask (最近邻插值, 保持标签值).

    shift_vec: (3,) 整数向量, 正=向正方向移动
    """
    sv = np.round(shift_vec).astype(int)
    if np.all(sv == 0):
        return mask.copy()

    result = np.zeros_like(mask)
    src_slices = []
    dst_slices = []
    for i in range(3):
        s = sv[i]
        dim = mask.shape[i]
        if s >= 0:
            src_slices.append(slice(0, dim - s))
            dst_slices.append(slice(s, dim))
        else:
            src_slices.append(slice(-s, dim))
            dst_slices.append(slice(0, dim + s))

    result[tuple(dst_slices)] = mask[tuple(src_slices)]
    return result


def fix_case(case_dir, crop_size=64, dry_run=False):
    """
    修复单个 case 的对齐.

    返回: dict with offset stats, or None if skipped
    """
    centroid_path = os.path.join(case_dir, "centroid.npy")
    mask_path = os.path.join(case_dir, "mask_aligned.nii.gz")

    if not os.path.exists(centroid_path) or not os.path.exists(mask_path):
        return None

    centroid_ras = np.load(centroid_path)
    mask = nib.load(mask_path).get_fdata().astype(np.int32)

    fg_centroid = compute_fg_centroid(mask)
    if fg_centroid is None:
        return None

    center = np.array([crop_size / 2] * 3)
    offset = fg_centroid - center

    if dry_run:
        return {
            "old_centroid": centroid_ras.tolist(),
            "fg_centroid_local": fg_centroid.tolist(),
            "offset": offset.tolist(),
            "offset_norm": float(np.linalg.norm(offset)),
        }

    # --- 备份原始文件 ---
    orig_centroid = os.path.join(case_dir, "centroid_orig.npy")
    orig_mask = os.path.join(case_dir, "mask_aligned_orig.nii.gz")
    if not os.path.exists(orig_centroid):
        shutil.copy2(centroid_path, orig_centroid)
    if not os.path.exists(orig_mask):
        shutil.copy2(mask_path, orig_mask)

    # --- 修复 centroid ---
    new_centroid = centroid_ras + offset
    np.save(centroid_path, new_centroid)

    # --- 修复 mask: 平移使前景居中 ---
    new_mask = shift_mask(mask, -offset)
    mask_nii = nib.Nifti1Image(new_mask.astype(np.uint8), np.eye(4))
    nib.save(mask_nii, mask_path)

    # --- 更新 info.json ---
    info_path = os.path.join(case_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        info["center_ras_orig"] = centroid_ras.tolist()
        info["center_ras"] = new_centroid.tolist()
        info["alignment_offset"] = offset.tolist()
        info["alignment_fixed"] = True
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    # 验证修复后的前景居中程度
    new_fg = compute_fg_centroid(new_mask)
    residual = np.linalg.norm(new_fg - center) if new_fg is not None else float('inf')

    return {
        "old_centroid": centroid_ras.tolist(),
        "new_centroid": new_centroid.tolist(),
        "fg_centroid_local": fg_centroid.tolist(),
        "offset": offset.tolist(),
        "offset_norm": float(np.linalg.norm(offset)),
        "residual_after_fix": float(residual),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅统计偏移, 不修改文件")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    prepared_dir = cfg["data"]["prepared_dir"]
    crop_size = cfg["segmentation"]["crop_size"]
    if isinstance(crop_size, list):
        crop_size = crop_size[0]

    case_dirs = sorted([
        d for d in os.listdir(prepared_dir)
        if os.path.isdir(os.path.join(prepared_dir, d)) and "_" in d
    ])

    print(f"{'DRY RUN: ' if args.dry_run else ''}修复 {len(case_dirs)} 个 case 的数据对齐")
    print(f"Prepared dir: {prepared_dir}")
    print(f"Crop size: {crop_size}")
    print()

    results = []
    for case_id in case_dirs:
        case_dir = os.path.join(prepared_dir, case_id)
        r = fix_case(case_dir, crop_size=crop_size, dry_run=args.dry_run)
        if r is not None:
            r["case_id"] = case_id
            results.append(r)

    if not results:
        print("没有找到可处理的 case")
        return

    # 统计
    offsets = np.array([r["offset"] for r in results])
    dists = np.array([r["offset_norm"] for r in results])

    print(f"\n{'='*60}")
    print(f"偏移统计 ({len(results)} cases):")
    print(f"  Mean |offset|:     {dists.mean():.2f} voxels")
    print(f"  Median |offset|:   {np.median(dists):.2f} voxels")
    print(f"  Max |offset|:      {dists.max():.2f} voxels")
    print(f"  Min |offset|:      {dists.min():.2f} voxels")
    print(f"  Per-axis mean:     {np.mean(np.abs(offsets), axis=0).round(2).tolist()}")
    print(f"  Per-axis std:      {np.std(offsets, axis=0).round(2).tolist()}")

    if not args.dry_run:
        residuals = [r["residual_after_fix"] for r in results]
        print(f"\n修复后残差:")
        print(f"  Mean residual:     {np.mean(residuals):.2f} voxels")
        print(f"  Max residual:      {np.max(residuals):.2f} voxels")
        print(f"\n原始文件已备份为 centroid_orig.npy, mask_aligned_orig.nii.gz")

    # 分布: 偏移大于阈值的 case
    for thr in [2, 3, 5]:
        n = (dists > thr).sum()
        print(f"  |offset| > {thr}: {n} cases ({100*n/len(results):.1f}%)")

    if args.dry_run:
        print(f"\n这是 DRY RUN, 未修改任何文件. 去掉 --dry_run 执行实际修复.")
    else:
        print(f"\n修复完成! 建议重新训练 Phase 2/3.")


if __name__ == "__main__":
    main()
