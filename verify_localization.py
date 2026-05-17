"""
verify_localization.py
=======================
LocSegNet Phase 1 定位模型 — 增强验证 & 可视化脚本.

在 visualize_prediction.py 基础上增加:
  1. 物理距离 (mm): 利用 NIfTI affine 将体素误差转换为真实毫米距离
  2. Heatmap 质量分析: 峰值强度、峰值锐度 (集中度)、GT-Peak 偏移
  3. 3D 散点图: 预测 vs GT 在体素空间的全局分布
  4. 置信度-误差相关性: heatmap 峰值置信度 vs 定位准确度
  5. 详细表格输出: 格式化 per-case 报告, 支持 CSV 导出
  6. NIfTI 输出: 保存 heatmap 为 NIfTI 方便 ITK-SNAP/3D Slicer 查看
  7. 综合仪表板: 单张大图汇总所有关键指标

用法:
    # 快速验证 (前 5 个 case, 含 mm 距离)
    python verify_localization.py --checkpoint checkpoints/phase1_epoch70.pth

    # 全验证集评估 + 完整报告
    python verify_localization.py --checkpoint checkpoints/phase1_epoch70.pth --eval_all

    # 指定 case 详细可视化
    python verify_localization.py --checkpoint checkpoints/phase1_epoch70.pth --case_id 01612716_ipsi

    # 导出 heatmap NIfTI
    python verify_localization.py --checkpoint checkpoints/phase1_epoch70.pth --save_nifti

    # 指定 GPU
    python verify_localization.py --checkpoint checkpoints/phase1_epoch70.pth --gpu_id 2
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import zoom as scipy_zoom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.locseg_net import LocSegNet
from data.dataset import TNLocSegDataset, get_train_val_split
from nnunet_loc_trainer import nnUNetTrainerLoc

# ----------------------------------------------------------------
# 自定义 colormaps
# ----------------------------------------------------------------
HEATMAP_COLORS = [
    (0.0, 0.0, 0.0, 0.0),
    (0.8, 0.0, 0.0, 0.4),
    (1.0, 0.2, 0.0, 0.6),
    (1.0, 0.6, 0.0, 0.8),
    (1.0, 1.0, 0.0, 0.95),
]
HEATMAP_CMAP = LinearSegmentedColormap.from_list("heatmap_overlay", HEATMAP_COLORS, N=256)


# ================================================================
# 模型加载 & 推理
# ================================================================

def load_model(cfg, checkpoint_path, device):
    """加载模型和 checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "network_weights" in ckpt:
        return _load_nnunet_loc_model(cfg, checkpoint_path, ckpt, device)
    return _load_locseg_model(cfg, checkpoint_path, ckpt, device)


def _load_locseg_model(cfg, checkpoint_path, ckpt, device):
    """加载 LocSegNet 风格 checkpoint."""
    seg_arch = cfg["segmentation"].get("seg_arch", "nnunet")
    model = LocSegNet(
        loc_input_size=tuple(cfg["localization"]["input_size"]),
        seg_crop_size=tuple(cfg["segmentation"]["crop_size"]),
        num_classes=cfg["segmentation"]["num_classes"],
        loc_channels=tuple(cfg["localization"]["encoder_channels"]),
        seg_channels=tuple(cfg["segmentation"].get("encoder_channels", [32, 64, 128, 256, 512])),
        deep_supervision=cfg["segmentation"]["deep_supervision"],
        seg_arch=seg_arch,
        seg_features_per_stage=tuple(cfg["segmentation"].get("features_per_stage", [32, 64, 128, 256, 320])),
        n_conv_per_stage=cfg["segmentation"].get("n_conv_per_stage", 2),
        n_conv_per_stage_decoder=cfg["segmentation"].get("n_conv_per_stage_decoder", 2),
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    phase = ckpt.get("phase", "?")
    print(f"[OK] 模型加载: {checkpoint_path} (phase={phase}, epoch={epoch}, dual-head)")
    return model, ckpt


class _NnUNetLocAdapter(torch.nn.Module):
    """将 nnUNetTrainerLoc 封装为与 LocSegNet forward_loc 一致的接口."""

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.net = trainer.network
        self.loc_input_size = tuple(getattr(trainer, "loc_input_size", (128, 128, 128)))

    def eval(self):
        self.net.eval()
        return self

    def forward_loc(self, wb, gt_centroid_norm=None):
        mod = self.net.module if hasattr(self.net, "module") else self.net
        loc_in = wb
        if tuple(wb.shape[2:]) != self.loc_input_size:
            loc_in = torch.nn.functional.interpolate(
                wb, size=self.loc_input_size, mode="trilinear", align_corners=False
            )
        hm_l, hm_r, c_l, c_r = mod.forward_loc(loc_in)
        if gt_centroid_norm is not None:
            use_left = (gt_centroid_norm[:, 0] < 0.5).view(-1, 1)
            c = torch.where(use_left, c_l, c_r)
            hm = torch.where((gt_centroid_norm[:, 0] < 0.5).view(-1, 1, 1, 1, 1), hm_l, hm_r)
        else:
            c = 0.5 * (c_l + c_r)
            hm = 0.5 * (hm_l + hm_r)
        return {"heatmap": hm, "centroid_norm": c}


def _load_nnunet_loc_model(cfg, checkpoint_path, ckpt, device):
    """加载 nnUNetTrainerLoc 风格 checkpoint."""
    init_args = ckpt["init_args"]
    trainer = nnUNetTrainerLoc(
        plans=init_args["plans"],
        configuration=init_args["configuration"],
        fold=init_args["fold"],
        dataset_json=init_args["dataset_json"],
        device=torch.device(device),
    )
    trainer.initialize()
    # 可视化只需要网络权重; 训练期我们改过 phase 优化器参数组,
    # 直接 load_checkpoint 会在 optimizer_state 上报 group 数不匹配.
    trainer.network.load_state_dict(ckpt["network_weights"])
    model = _NnUNetLocAdapter(trainer).to(device).eval()
    epoch = ckpt.get("current_epoch", "?")
    print(f"[OK] 模型加载: {checkpoint_path} (nnUNetTrainerLoc, epoch={epoch})")
    return model, ckpt


def predict_single(model, wb_tensor, device, gt_centroid_norm=None):
    """
    对单个 whole brain 做定位预测, 返回 heatmap + centroid.

    gt_centroid_norm: (3,) numpy array, 用于选择 dual-head 中的正确 head.
    """
    model.eval()
    with torch.no_grad():
        wb = wb_tensor.unsqueeze(0).to(device)

        # 传入 GT centroid 用于选择 head
        if gt_centroid_norm is not None:
            gt = torch.from_numpy(gt_centroid_norm).unsqueeze(0).float().to(device)
        else:
            gt = None

        loc_result = model.forward_loc(wb, gt_centroid_norm=gt)

    hm_np = loc_result["heatmap"].cpu().squeeze(0).squeeze(0).numpy()
    cn_np = loc_result["centroid_norm"].cpu().squeeze(0).numpy()
    return {"centroid_norm": cn_np, "heatmap": hm_np}


# ================================================================
# 物理距离计算
# ================================================================

def get_voxel_spacing(nii_path):
    """从 NIfTI header 获取体素物理尺寸 (mm)."""
    nii = nib.load(nii_path)
    spacing = nii.header.get_zooms()[:3]
    return np.array(spacing, dtype=np.float32)


def voxel_to_mm(voxel_error, spacing):
    """将体素误差转换为毫米误差."""
    return np.sqrt(np.sum((voxel_error * spacing) ** 2))


def per_axis_voxel_to_mm(per_axis_vox, spacing):
    """逐轴体素误差转为毫米."""
    return per_axis_vox * spacing


# ================================================================
# Heatmap 质量分析
# ================================================================

def analyze_heatmap(heatmap, gt_norm, loc_input_size):
    """
    分析 heatmap 的质量指标:
      - peak_value: heatmap 最大值 (越高越自信)
      - peak_loc_norm: 峰值位置 (归一化)
      - peak_gt_dist_vox: 峰值位置与 GT 在低分辨率空间的距离
      - sharpness: 集中度 = top-1%体素均值 / 全体均值 (越大越尖锐)
      - entropy: heatmap 归一化后的信息熵 (越小越集中)
    """
    hm_shape = np.array(loc_input_size, dtype=np.float32)

    # 峰值
    peak_value = float(heatmap.max())
    peak_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    peak_loc_norm = np.array(peak_idx, dtype=np.float32) / (hm_shape - 1)

    # 峰值 vs GT 距离 (低分辨率空间体素)
    gt_vox_lr = gt_norm * (hm_shape - 1)
    peak_vox_lr = np.array(peak_idx, dtype=np.float32)
    peak_gt_dist_vox = float(np.linalg.norm(peak_vox_lr - gt_vox_lr))

    # 锐度 (top-1% 均值 / 整体均值)
    flat = heatmap.flatten()
    top_k = max(1, int(len(flat) * 0.01))
    top_vals = np.partition(flat, -top_k)[-top_k:]
    mean_all = flat.mean()
    sharpness = float(top_vals.mean() / (mean_all + 1e-10))

    # 信息熵
    hm_pos = np.clip(heatmap, 0, None)
    hm_prob = hm_pos / (hm_pos.sum() + 1e-10)
    hm_prob_flat = hm_prob.flatten()
    hm_prob_flat = hm_prob_flat[hm_prob_flat > 1e-12]
    entropy = float(-np.sum(hm_prob_flat * np.log(hm_prob_flat)))

    return {
        "peak_value": peak_value,
        "peak_loc_norm": peak_loc_norm,
        "peak_gt_dist_vox": peak_gt_dist_vox,
        "sharpness": sharpness,
        "entropy": entropy,
    }


# ================================================================
# 核心评估流程
# ================================================================

def evaluate_all(model, val_dataset, device, loc_input_size, cfg):
    """在验证集上运行推理, 返回详细结果列表."""
    results = []
    for idx in range(len(val_dataset)):
        sample = val_dataset[idx]
        case_id = sample["case_id"]
        wb_tensor = sample["whole_brain"]
        gt_norm = sample["centroid_norm"].numpy()
        wb_shape = sample["wb_shape"].numpy()

        # 获取物理 spacing
        case_info = val_dataset.cases[idx]
        spacing = get_voxel_spacing(case_info["nii_path"])

        # 推理 (传入 GT centroid 用于选择 dual-head)
        pred = predict_single(model, wb_tensor, device, gt_centroid_norm=gt_norm)
        pred_norm = pred["centroid_norm"]
        heatmap = pred["heatmap"]

        # 体素坐标
        gt_vox = gt_norm * (wb_shape - 1)
        pred_vox = pred_norm * (wb_shape - 1)
        error_vox = float(np.linalg.norm(gt_vox - pred_vox))
        per_axis_err_vox = np.abs(gt_vox - pred_vox)

        # 物理距离 (mm)
        per_axis_err_mm = per_axis_voxel_to_mm(per_axis_err_vox, spacing)
        error_mm = voxel_to_mm(per_axis_err_vox, spacing)

        # Heatmap 质量
        hm_analysis = analyze_heatmap(heatmap, gt_norm, loc_input_size)

        # 解析 side
        side = case_id.rsplit("_", 1)[-1] if "_" in case_id else "unknown"

        results.append({
            "idx": idx,
            "case_id": case_id,
            "side": side,
            "nii_path": case_info["nii_path"],
            "spacing": spacing,
            "gt_norm": gt_norm,
            "pred_norm": pred_norm,
            "gt_vox": gt_vox,
            "pred_vox": pred_vox,
            "wb_shape": wb_shape,
            "error_vox": error_vox,
            "error_mm": error_mm,
            "per_axis_err_vox": per_axis_err_vox,
            "per_axis_err_mm": per_axis_err_mm,
            "heatmap": heatmap,
            **hm_analysis,
        })

        print(f"  [{idx+1}/{len(val_dataset)}] {case_id} ({side}): "
              f"error = {error_vox:.1f} vox / {error_mm:.2f} mm  |  "
              f"peak={hm_analysis['peak_value']:.2f}  sharp={hm_analysis['sharpness']:.1f}")

    return results


# ================================================================
# 可视化函数
# ================================================================

def _normalize_image(img):
    """归一化到 [0, 1] 用于显示."""
    if (img > 0).any():
        vmin, vmax = np.percentile(img[img > 0], [1, 99])
    else:
        vmin, vmax = img.min(), img.max()
    return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)


def visualize_single_case(wb_raw, gt_vox, pred_vox, heatmap, case_id,
                          output_dir, loc_input_size, error_vox, error_mm,
                          spacing, hm_analysis):
    """
    单个 case 的综合可视化 (4 行 3 列):
      行1: MRI 三视图 + GT/Pred centroid 标注
      行2: MRI + Heatmap 叠加 (alpha blend)
      行3: Centroid 局部放大 (zoom in +/-30 voxels)
      行4: Heatmap 等高线 + 峰值分析
    """
    fig = plt.figure(figsize=(22, 26))
    gs = GridSpec(4, 3, figure=fig, hspace=0.28, wspace=0.15)

    fig.suptitle(
        f"Case: {case_id}\n"
        f"GT: [{gt_vox[0]:.0f}, {gt_vox[1]:.0f}, {gt_vox[2]:.0f}]   "
        f"Pred: [{pred_vox[0]:.1f}, {pred_vox[1]:.1f}, {pred_vox[2]:.1f}]   "
        f"Error: {error_vox:.1f} vox / {error_mm:.2f} mm\n"
        f"Spacing: [{spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f}] mm   "
        f"Peak: {hm_analysis['peak_value']:.3f}   "
        f"Sharpness: {hm_analysis['sharpness']:.1f}   "
        f"Peak-GT dist: {hm_analysis['peak_gt_dist_vox']:.1f} vox(LR)",
        fontsize=13, fontweight="bold", y=0.99,
    )

    wb = _normalize_image(wb_raw)
    gt = gt_vox.astype(int)
    pred = pred_vox

    view_names = ["Sagittal (dim0)", "Coronal (dim1)", "Axial (dim2)"]

    # 三视图切片 (在 GT centroid 所在层)
    slices_wb = [
        wb[np.clip(gt[0], 0, wb.shape[0] - 1), :, :],
        wb[:, np.clip(gt[1], 0, wb.shape[1] - 1), :],
        wb[:, :, np.clip(gt[2], 0, wb.shape[2] - 1)],
    ]
    markers = [
        ((gt[2], gt[1]), (pred[2], pred[1])),
        ((gt[2], gt[0]), (pred[2], pred[0])),
        ((gt[1], gt[0]), (pred[1], pred[0])),
    ]

    # ---- 行1: MRI 三视图 + centroid ----
    for col in range(3):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(slices_wb[col], cmap="gray", origin="lower", aspect="auto")
        gt_xy, pred_xy = markers[col]
        ax.plot(*gt_xy, "r+", markersize=18, markeredgewidth=2.5, label="GT")
        ax.plot(*pred_xy, "c*", markersize=14, markeredgewidth=1.5, label="Pred")
        ax.plot([gt_xy[0], pred_xy[0]], [gt_xy[1], pred_xy[1]],
                "y--", linewidth=1.2, alpha=0.7)
        ax.set_title(f"{view_names[col]} (slice={gt[col]})", fontsize=11)
        if col == 0:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
        ax.axis("off")

    # ---- 行2: Heatmap 叠加 ----
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    wb_shape = np.array(wb_raw.shape[:3], dtype=np.float32)
    hm_shape = np.array(loc_input_size, dtype=np.float32)
    gt_hm = (gt_vox / (wb_shape - 1) * (hm_shape - 1)).astype(int)
    pred_hm = pred_vox / (wb_shape - 1) * (hm_shape - 1)

    scale = hm_shape / wb_shape
    wb_low = scipy_zoom(wb, scale, order=1)

    slices_hm_wb = [
        wb_low[np.clip(gt_hm[0], 0, wb_low.shape[0] - 1), :, :],
        wb_low[:, np.clip(gt_hm[1], 0, wb_low.shape[1] - 1), :],
        wb_low[:, :, np.clip(gt_hm[2], 0, wb_low.shape[2] - 1)],
    ]
    slices_hm = [
        heatmap_norm[np.clip(gt_hm[0], 0, heatmap_norm.shape[0] - 1), :, :],
        heatmap_norm[:, np.clip(gt_hm[1], 0, heatmap_norm.shape[1] - 1), :],
        heatmap_norm[:, :, np.clip(gt_hm[2], 0, heatmap_norm.shape[2] - 1)],
    ]
    markers_hm = [
        ((gt_hm[2], gt_hm[1]), (pred_hm[2], pred_hm[1])),
        ((gt_hm[2], gt_hm[0]), (pred_hm[2], pred_hm[0])),
        ((gt_hm[1], gt_hm[0]), (pred_hm[1], pred_hm[0])),
    ]

    for col in range(3):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(slices_hm_wb[col], cmap="gray", origin="lower", aspect="auto")
        ax.imshow(slices_hm[col], cmap=HEATMAP_CMAP, origin="lower", aspect="auto",
                  vmin=0, vmax=1)
        gt_xy, pred_xy = markers_hm[col]
        ax.plot(*gt_xy, "g+", markersize=16, markeredgewidth=2.5, label="GT")
        ax.plot(*pred_xy, "c*", markersize=12, markeredgewidth=1.5, label="Pred")
        ax.set_title(f"Heatmap Overlay - {view_names[col]}", fontsize=11)
        if col == 0:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
        ax.axis("off")

    # ---- 行3: Centroid 局部放大 (+/-30 voxels) ----
    zoom_radius = 30
    for col in range(3):
        ax = fig.add_subplot(gs[2, col])
        sl = slices_wb[col]
        gt_xy, pred_xy = markers[col]
        cx, cy = int(gt_xy[0]), int(gt_xy[1])
        x0, x1 = max(0, cx - zoom_radius), min(sl.shape[1], cx + zoom_radius)
        y0, y1 = max(0, cy - zoom_radius), min(sl.shape[0], cy + zoom_radius)
        crop = sl[y0:y1, x0:x1]

        ax.imshow(crop, cmap="gray", origin="lower", aspect="auto",
                  extent=[x0, x1, y0, y1])
        ax.plot(*gt_xy, "r+", markersize=22, markeredgewidth=3, label="GT")
        ax.plot(*pred_xy, "c*", markersize=18, markeredgewidth=2, label="Pred")
        ax.plot([gt_xy[0], pred_xy[0]], [gt_xy[1], pred_xy[1]],
                "y--", linewidth=1.5, alpha=0.8)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(f"Zoom-in ({view_names[col]})", fontsize=11)
        if col == 0:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
        ax.axis("off")

    # ---- 行4: Heatmap 等高线图 (低分辨率空间) ----
    for col in range(3):
        ax = fig.add_subplot(gs[3, col])
        hm_slice = slices_hm[col]
        ax.imshow(slices_hm_wb[col], cmap="gray", origin="lower", aspect="auto", alpha=0.5)
        # 等高线
        levels = np.linspace(0.1, 0.9, 9)
        contour = ax.contour(hm_slice, levels=levels, cmap="hot",
                             origin="lower", linewidths=0.8)
        ax.clabel(contour, inline=True, fontsize=6, fmt="%.1f")
        # 标记
        gt_xy, pred_xy = markers_hm[col]
        ax.plot(*gt_xy, "g+", markersize=16, markeredgewidth=2.5, label="GT")
        ax.plot(*pred_xy, "c*", markersize=12, markeredgewidth=1.5, label="Pred")
        ax.set_title(f"Heatmap Contour - {view_names[col]}", fontsize=11)
        if col == 0:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
        ax.axis("off")

    save_path = os.path.join(output_dir, f"{case_id}_verify.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    -> 保存: {save_path}")
    return save_path


def plot_comprehensive_dashboard(results, output_dir, ckpt_name=""):
    """
    综合仪表板 (3x3 大图):
      (0,0) 误差直方图 (vox + mm 双轴)
      (0,1) CDF 曲线 (mm)
      (0,2) Ipsi vs Contra 箱线图 (mm)
      (1,0) Per-axis 误差 (mm)
      (1,1) 置信度 vs 误差 散点图
      (1,2) 锐度 vs 误差 散点图
      (2,0) Peak-GT 距离 vs 误差
      (2,1) Heatmap 熵 vs 误差
      (2,2) 3D 散点图 (GT vs Pred)
    """
    errors_vox = np.array([r["error_vox"] for r in results])
    errors_mm = np.array([r["error_mm"] for r in results])
    per_axis_mm = np.array([r["per_axis_err_mm"] for r in results])
    per_axis_vox = np.array([r["per_axis_err_vox"] for r in results])
    peak_values = np.array([r["peak_value"] for r in results])
    sharpness_vals = np.array([r["sharpness"] for r in results])
    peak_gt_dists = np.array([r["peak_gt_dist_vox"] for r in results])
    entropies = np.array([r["entropy"] for r in results])
    sides = [r["side"] for r in results]

    fig = plt.figure(figsize=(24, 22))
    gs = GridSpec(3, 3, figure=fig, hspace=0.32, wspace=0.28)
    fig.suptitle(
        f"Localization Verification Dashboard  |  {len(results)} cases  |  {ckpt_name}\n"
        f"Mean: {errors_vox.mean():.1f} vox / {errors_mm.mean():.2f} mm   "
        f"Median: {np.median(errors_vox):.1f} vox / {np.median(errors_mm):.2f} mm",
        fontsize=15, fontweight="bold", y=0.99,
    )

    # ---- (0,0) 误差直方图 (vox + mm) ----
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(errors_vox, bins=25, color="steelblue", edgecolor="white", alpha=0.85,
            label="Voxel")
    ax.axvline(errors_vox.mean(), color="red", ls="--", lw=2,
               label=f"Mean: {errors_vox.mean():.1f} vox")
    ax.axvline(np.median(errors_vox), color="orange", ls="--", lw=2,
               label=f"Median: {np.median(errors_vox):.1f} vox")
    ax.set_xlabel("Localization Error (voxels)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Error Distribution (Voxel)", fontsize=12)
    ax.legend(fontsize=9)
    # 副轴: mm
    ax2 = ax.twiny()
    ax2.hist(errors_mm, bins=25, alpha=0)  # 不可见, 仅为了设置 x 轴
    ax2.set_xlabel("(mm)", fontsize=9, color="gray")
    ax2.tick_params(axis="x", labelsize=8, colors="gray")

    # ---- (0,1) CDF 曲线 (mm) ----
    ax = fig.add_subplot(gs[0, 1])
    sorted_mm = np.sort(errors_mm)
    cdf = np.arange(1, len(sorted_mm) + 1) / len(sorted_mm) * 100
    ax.plot(sorted_mm, cdf, "b-", linewidth=2)
    ax.fill_between(sorted_mm, 0, cdf, alpha=0.15, color="blue")
    for thr_mm, color in [(2.0, "green"), (5.0, "orange"), (10.0, "red")]:
        pct = (errors_mm < thr_mm).mean() * 100
        ax.axvline(thr_mm, color=color, ls=":", lw=1.5, alpha=0.7)
        ax.annotate(f"<{thr_mm}mm: {pct:.0f}%", xy=(thr_mm, pct),
                    fontsize=9, xytext=(thr_mm + 0.3, max(pct - 8, 5)),
                    color=color, fontweight="bold")
    ax.set_xlabel("Error (mm)", fontsize=11)
    ax.set_ylabel("Cumulative %", fontsize=11)
    ax.set_title("CDF (Physical Distance)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # ---- (0,2) Ipsi vs Contra 箱线图 (mm) ----
    ax = fig.add_subplot(gs[0, 2])
    ipsi_mm = [r["error_mm"] for r in results if r["side"] == "ipsi"]
    contra_mm = [r["error_mm"] for r in results if r["side"] == "contra"]
    data_box, labels_box = [], []
    if ipsi_mm:
        data_box.append(ipsi_mm)
        labels_box.append(f"Ipsi (n={len(ipsi_mm)})\n"
                          f"Mean={np.mean(ipsi_mm):.2f}mm")
    if contra_mm:
        data_box.append(contra_mm)
        labels_box.append(f"Contra (n={len(contra_mm)})\n"
                          f"Mean={np.mean(contra_mm):.2f}mm")
    if data_box:
        bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True, widths=0.5)
        colors_box = ["#3498db", "#e74c3c"]
        for patch, c in zip(bp["boxes"], colors_box[:len(data_box)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
    ax.set_ylabel("Error (mm)", fontsize=11)
    ax.set_title("Ipsi vs Contra (mm)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # ---- (1,0) Per-axis 误差 (mm + vox) ----
    ax = fig.add_subplot(gs[1, 0])
    axis_names = ["Dim0\n(L-R)", "Dim1\n(P-A)", "Dim2\n(I-S)"]
    means_mm = per_axis_mm.mean(axis=0)
    stds_mm = per_axis_mm.std(axis=0)
    means_vox = per_axis_vox.mean(axis=0)
    x = np.arange(3)
    width = 0.35
    bars1 = ax.bar(x - width / 2, means_vox, width, label="Voxel",
                   color=["#2ecc71", "#3498db", "#e67e22"], alpha=0.6,
                   edgecolor="white")
    bars2 = ax.bar(x + width / 2, means_mm, width, yerr=stds_mm, capsize=5,
                   label="mm", color=["#27ae60", "#2980b9", "#d35400"], alpha=0.9,
                   edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(axis_names, fontsize=10)
    ax.set_ylabel("Absolute Error", fontsize=11)
    ax.set_title("Per-axis Error (Voxel & mm)", fontsize=12)
    for bar, m in zip(bars2, means_mm):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{m:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ---- (1,1) 置信度 (Peak Value) vs 误差 ----
    ax = fig.add_subplot(gs[1, 1])
    sc = ax.scatter(peak_values, errors_mm, c=errors_mm, cmap="RdYlGn_r",
                    s=40, alpha=0.7, edgecolors="gray", linewidths=0.5)
    # 线性回归
    if len(peak_values) > 2:
        z = np.polyfit(peak_values, errors_mm, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(peak_values.min(), peak_values.max(), 50)
        ax.plot(x_fit, p(x_fit), "r--", lw=1.5, alpha=0.7)
        corr = np.corrcoef(peak_values, errors_mm)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=10, va="top", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.colorbar(sc, ax=ax, label="Error (mm)", shrink=0.8)
    ax.set_xlabel("Heatmap Peak Value", fontsize=11)
    ax.set_ylabel("Error (mm)", fontsize=11)
    ax.set_title("Confidence vs Error", fontsize=12)
    ax.grid(True, alpha=0.3)

    # ---- (1,2) 锐度 vs 误差 ----
    ax = fig.add_subplot(gs[1, 2])
    sc = ax.scatter(sharpness_vals, errors_mm, c=errors_mm, cmap="RdYlGn_r",
                    s=40, alpha=0.7, edgecolors="gray", linewidths=0.5)
    if len(sharpness_vals) > 2:
        z = np.polyfit(sharpness_vals, errors_mm, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(sharpness_vals.min(), sharpness_vals.max(), 50)
        ax.plot(x_fit, p(x_fit), "r--", lw=1.5, alpha=0.7)
        corr = np.corrcoef(sharpness_vals, errors_mm)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=10, va="top", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.colorbar(sc, ax=ax, label="Error (mm)", shrink=0.8)
    ax.set_xlabel("Heatmap Sharpness (top1% / mean)", fontsize=11)
    ax.set_ylabel("Error (mm)", fontsize=11)
    ax.set_title("Sharpness vs Error", fontsize=12)
    ax.grid(True, alpha=0.3)

    # ---- (2,0) Peak-GT 距离 vs 预测误差 ----
    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(peak_gt_dists, errors_vox, c="steelblue", s=40, alpha=0.7,
               edgecolors="gray", linewidths=0.5)
    # 对角线参考
    max_val = max(peak_gt_dists.max(), errors_vox.max())
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, alpha=0.4, label="y=x")
    if len(peak_gt_dists) > 2:
        corr = np.corrcoef(peak_gt_dists, errors_vox)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=10, va="top", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set_xlabel("Peak-GT Distance (vox, low-res)", fontsize=11)
    ax.set_ylabel("Prediction Error (vox, full-res)", fontsize=11)
    ax.set_title("Peak Location Accuracy", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- (2,1) Heatmap 熵 vs 误差 ----
    ax = fig.add_subplot(gs[2, 1])
    sc = ax.scatter(entropies, errors_mm, c=sharpness_vals, cmap="viridis",
                    s=40, alpha=0.7, edgecolors="gray", linewidths=0.5)
    if len(entropies) > 2:
        corr = np.corrcoef(entropies, errors_mm)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=10, va="top", fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.colorbar(sc, ax=ax, label="Sharpness", shrink=0.8)
    ax.set_xlabel("Heatmap Entropy", fontsize=11)
    ax.set_ylabel("Error (mm)", fontsize=11)
    ax.set_title("Entropy vs Error", fontsize=12)
    ax.grid(True, alpha=0.3)

    # ---- (2,2) 3D 散点图: 预测偏移方向 ----
    ax = fig.add_subplot(gs[2, 2], projection="3d")
    gt_all = np.array([r["gt_vox"] for r in results])
    pred_all = np.array([r["pred_vox"] for r in results])
    offsets = pred_all - gt_all  # 偏移向量

    sc = ax.scatter(offsets[:, 0], offsets[:, 1], offsets[:, 2],
                    c=errors_mm, cmap="RdYlGn_r", s=30, alpha=0.7,
                    edgecolors="gray", linewidths=0.3)
    # 原点
    ax.scatter([0], [0], [0], c="black", s=100, marker="x", linewidths=2,
               label="Origin (GT)")
    ax.set_xlabel("Dim0 offset (vox)", fontsize=9)
    ax.set_ylabel("Dim1 offset (vox)", fontsize=9)
    ax.set_zlabel("Dim2 offset (vox)", fontsize=9)
    ax.set_title("Prediction Offset in 3D", fontsize=12)
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, label="Error (mm)", shrink=0.6, pad=0.12)

    save_path = os.path.join(output_dir, "verification_dashboard.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    -> 保存: {save_path}")
    return save_path


def plot_per_case_ranking(results, output_dir):
    """Per-case 排名图 (mm 为主, vox 为辅)."""
    errors_mm = np.array([r["error_mm"] for r in results])
    errors_vox = np.array([r["error_vox"] for r in results])
    case_ids = [r["case_id"] for r in results]
    sides_list = [r["side"] for r in results]

    sorted_idx = np.argsort(errors_mm)[::-1]
    n_show = min(50, len(sorted_idx))

    fig, ax = plt.subplots(figsize=(12, max(6, n_show * 0.25)))
    colors_bar = []
    for i in range(n_show):
        e = errors_mm[sorted_idx[i]]
        if e > 10:
            colors_bar.append("#e74c3c")
        elif e > 5:
            colors_bar.append("#f39c12")
        else:
            colors_bar.append("#2ecc71")

    bars = ax.barh(range(n_show),
                   [errors_mm[sorted_idx[i]] for i in range(n_show)],
                   color=colors_bar, edgecolor="white")
    ax.set_yticks(range(n_show))
    labels = [
        f"{case_ids[sorted_idx[i]]} ({sides_list[sorted_idx[i]]}) "
        f"[{errors_vox[sorted_idx[i]]:.0f}vox]"
        for i in range(n_show)
    ]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Error (mm)", fontsize=12)
    ax.set_title(f"Per-case Error Ranking (top {n_show}, mm)", fontsize=13)
    ax.invert_yaxis()
    legend_patches = [
        mpatches.Patch(color="#e74c3c", label="> 10 mm"),
        mpatches.Patch(color="#f39c12", label="5-10 mm"),
        mpatches.Patch(color="#2ecc71", label="< 5 mm"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "error_ranking_mm.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    -> 保存: {save_path}")
    return save_path


# ================================================================
# 报告输出
# ================================================================

def print_detailed_report(results):
    """打印详细统计报告."""
    errors_vox = np.array([r["error_vox"] for r in results])
    errors_mm = np.array([r["error_mm"] for r in results])
    per_axis_vox = np.array([r["per_axis_err_vox"] for r in results])
    per_axis_mm = np.array([r["per_axis_err_mm"] for r in results])
    peak_vals = np.array([r["peak_value"] for r in results])
    sharpness = np.array([r["sharpness"] for r in results])

    w = 60
    print(f"\n{'=' * w}")
    print(f"  定位验证报告  ({len(results)} cases)")
    print(f"{'=' * w}")

    print(f"\n  --- 体素距离 (voxels) ---")
    print(f"  Mean:    {errors_vox.mean():>8.2f}")
    print(f"  Median:  {np.median(errors_vox):>8.2f}")
    print(f"  Std:     {errors_vox.std():>8.2f}")
    print(f"  Min:     {errors_vox.min():>8.2f}")
    print(f"  Max:     {errors_vox.max():>8.2f}")

    print(f"\n  --- 物理距离 (mm) ---")
    print(f"  Mean:    {errors_mm.mean():>8.2f}")
    print(f"  Median:  {np.median(errors_mm):>8.2f}")
    print(f"  Std:     {errors_mm.std():>8.2f}")
    print(f"  Min:     {errors_mm.min():>8.2f}")
    print(f"  Max:     {errors_mm.max():>8.2f}")

    print(f"\n  --- 阈值分析 (mm) ---")
    for thr in [2.0, 3.0, 5.0, 10.0]:
        n = (errors_mm < thr).sum()
        pct = (errors_mm < thr).mean() * 100
        print(f"  < {thr:>5.1f} mm: {n:>3}/{len(errors_mm)} ({pct:>5.1f}%)")

    print(f"\n  --- 阈值分析 (voxels) ---")
    for thr in [10, 20, 30, 50]:
        n = (errors_vox < thr).sum()
        pct = (errors_vox < thr).mean() * 100
        print(f"  < {thr:>3} vox:  {n:>3}/{len(errors_vox)} ({pct:>5.1f}%)")

    print(f"\n  --- Per-axis (mm) ---")
    axis_names = ["Dim0 (L-R)", "Dim1 (P-A)", "Dim2 (I-S)"]
    for i, name in enumerate(axis_names):
        print(f"  {name}: mean={per_axis_mm[:, i].mean():.2f}, "
              f"std={per_axis_mm[:, i].std():.2f}, "
              f"max={per_axis_mm[:, i].max():.2f}")

    print(f"\n  --- Per-axis (voxels) ---")
    for i, name in enumerate(axis_names):
        print(f"  {name}: mean={per_axis_vox[:, i].mean():.2f}, "
              f"std={per_axis_vox[:, i].std():.2f}, "
              f"max={per_axis_vox[:, i].max():.2f}")

    # Ipsi / Contra
    ipsi = [r for r in results if r["side"] == "ipsi"]
    contra = [r for r in results if r["side"] == "contra"]
    if ipsi:
        ipsi_mm = np.array([r["error_mm"] for r in ipsi])
        print(f"\n  --- Ipsi (n={len(ipsi)}) ---")
        print(f"  Mean: {ipsi_mm.mean():.2f} mm, Median: {np.median(ipsi_mm):.2f} mm")
    if contra:
        contra_mm = np.array([r["error_mm"] for r in contra])
        print(f"\n  --- Contra (n={len(contra)}) ---")
        print(f"  Mean: {contra_mm.mean():.2f} mm, Median: {np.median(contra_mm):.2f} mm")

    # Heatmap 质量
    print(f"\n  --- Heatmap 质量 ---")
    print(f"  Peak Value:  mean={peak_vals.mean():.3f}, std={peak_vals.std():.3f}")
    print(f"  Sharpness:   mean={sharpness.mean():.1f}, std={sharpness.std():.1f}")

    # 置信度-误差相关性
    if len(peak_vals) > 2:
        corr_peak = np.corrcoef(peak_vals, errors_mm)[0, 1]
        corr_sharp = np.corrcoef(sharpness, errors_mm)[0, 1]
        print(f"  Peak-Error corr:      r = {corr_peak:.3f}")
        print(f"  Sharpness-Error corr: r = {corr_sharp:.3f}")

    print(f"{'=' * w}\n")


def export_csv(results, output_dir):
    """导出 per-case 结果为 CSV."""
    csv_path = os.path.join(output_dir, "verification_results.csv")
    fieldnames = [
        "case_id", "side",
        "error_vox", "error_mm",
        "err_dim0_vox", "err_dim1_vox", "err_dim2_vox",
        "err_dim0_mm", "err_dim1_mm", "err_dim2_mm",
        "gt_d", "gt_h", "gt_w",
        "pred_d", "pred_h", "pred_w",
        "spacing_d", "spacing_h", "spacing_w",
        "peak_value", "sharpness", "entropy", "peak_gt_dist_vox",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "case_id": r["case_id"],
                "side": r["side"],
                "error_vox": f"{r['error_vox']:.2f}",
                "error_mm": f"{r['error_mm']:.3f}",
                "err_dim0_vox": f"{r['per_axis_err_vox'][0]:.2f}",
                "err_dim1_vox": f"{r['per_axis_err_vox'][1]:.2f}",
                "err_dim2_vox": f"{r['per_axis_err_vox'][2]:.2f}",
                "err_dim0_mm": f"{r['per_axis_err_mm'][0]:.3f}",
                "err_dim1_mm": f"{r['per_axis_err_mm'][1]:.3f}",
                "err_dim2_mm": f"{r['per_axis_err_mm'][2]:.3f}",
                "gt_d": f"{r['gt_vox'][0]:.1f}",
                "gt_h": f"{r['gt_vox'][1]:.1f}",
                "gt_w": f"{r['gt_vox'][2]:.1f}",
                "pred_d": f"{r['pred_vox'][0]:.1f}",
                "pred_h": f"{r['pred_vox'][1]:.1f}",
                "pred_w": f"{r['pred_vox'][2]:.1f}",
                "spacing_d": f"{r['spacing'][0]:.4f}",
                "spacing_h": f"{r['spacing'][1]:.4f}",
                "spacing_w": f"{r['spacing'][2]:.4f}",
                "peak_value": f"{r['peak_value']:.4f}",
                "sharpness": f"{r['sharpness']:.2f}",
                "entropy": f"{r['entropy']:.4f}",
                "peak_gt_dist_vox": f"{r['peak_gt_dist_vox']:.2f}",
            })
    print(f"    -> CSV 导出: {csv_path}")
    return csv_path


def save_heatmap_nifti(results, val_dataset, output_dir, loc_input_size):
    """将预测 heatmap 保存为 NIfTI 文件 (方便在 ITK-SNAP 中查看)."""
    nifti_dir = os.path.join(output_dir, "heatmap_nifti")
    os.makedirs(nifti_dir, exist_ok=True)

    for r in results:
        case_id = r["case_id"]
        heatmap = r["heatmap"]
        nii_path = r["nii_path"]

        # 加载原始 NIfTI 获取 affine
        ref_nii = nib.load(nii_path)
        ref_affine = ref_nii.affine.copy()
        ref_shape = np.array(ref_nii.header.get_data_shape()[:3], dtype=np.float32)
        hm_shape = np.array(loc_input_size, dtype=np.float32)

        # 调整 affine 以匹配低分辨率 heatmap
        scale = ref_shape / hm_shape
        hm_affine = ref_affine.copy()
        hm_affine[:3, :3] = ref_affine[:3, :3] @ np.diag(scale)

        # 归一化 heatmap 到 [0, 1]
        hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        hm_nii = nib.Nifti1Image(hm_norm.astype(np.float32), hm_affine)

        save_path = os.path.join(nifti_dir, f"{case_id}_heatmap.nii.gz")
        nib.save(hm_nii, save_path)

    print(f"    -> NIfTI heatmaps 保存到: {nifti_dir}/ ({len(results)} files)")


# ================================================================
# 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LocSegNet 定位模型增强验证 & 可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint 路径")
    parser.add_argument("--output_dir", type=str, default="verify_results",
                        help="输出目录")
    parser.add_argument("--num_cases", type=int, default=5,
                        help="可视化 case 数量 (非 eval_all 模式)")
    parser.add_argument("--case_id", type=str, default=None,
                        help="指定单个 case id")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="GPU 编号")
    parser.add_argument("--eval_all", action="store_true",
                        help="评估全部验证集")
    parser.add_argument("--save_nifti", action="store_true",
                        help="将 heatmap 保存为 NIfTI 文件")
    parser.add_argument("--no_vis", action="store_true",
                        help="跳过单例可视化, 仅生成统计报告")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # --- 设备 ---
    device_cfg = cfg.get("device", {})
    gpu_id = args.gpu_id if args.gpu_id is not None else device_cfg.get("gpu_id", 0)
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
    else:
        device = "cpu"
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    loc_input_size = tuple(cfg["localization"]["input_size"])

    # --- 加载模型 ---
    model, ckpt = load_model(cfg, args.checkpoint, device)
    ckpt_name = os.path.basename(args.checkpoint)

    # --- 准备验证集 ---
    prepared_dir = cfg["data"]["prepared_dir"]
    _, val_ids = get_train_val_split(
        prepared_dir,
        train_ratio=cfg["data"]["train_val_split"],
        seed=cfg["data"]["random_seed"],
    )
    val_dataset = TNLocSegDataset(prepared_dir, case_ids=val_ids, phase=1)
    print(f"验证集: {len(val_dataset)} cases")

    # --- 确定评估范围 ---
    if args.case_id:
        target_indices = [
            i for i, c in enumerate(val_dataset.cases)
            if c["case_id"] == args.case_id
        ]
        if not target_indices:
            print(f"[错误] '{args.case_id}' 不在验证集中.")
            print(f"可选 (前 20 个): {[c['case_id'] for c in val_dataset.cases[:20]]}")
            return
    elif args.eval_all:
        target_indices = list(range(len(val_dataset)))
    else:
        target_indices = list(range(min(args.num_cases, len(val_dataset))))

    # --- 推理 ---
    print(f"\n开始推理 ({len(target_indices)} cases)...")
    all_results = []
    for idx in target_indices:
        sample = val_dataset[idx]
        case_id = sample["case_id"]
        wb_tensor = sample["whole_brain"]
        gt_norm = sample["centroid_norm"].numpy()
        wb_shape = sample["wb_shape"].numpy()

        case_info = val_dataset.cases[idx]
        spacing = get_voxel_spacing(case_info["nii_path"])

        pred = predict_single(model, wb_tensor, device, gt_centroid_norm=gt_norm)
        pred_norm = pred["centroid_norm"]
        heatmap = pred["heatmap"]

        gt_vox = gt_norm * (wb_shape - 1)
        pred_vox = pred_norm * (wb_shape - 1)
        error_vox = float(np.linalg.norm(gt_vox - pred_vox))
        per_axis_err_vox = np.abs(gt_vox - pred_vox)
        per_axis_err_mm = per_axis_voxel_to_mm(per_axis_err_vox, spacing)
        error_mm = voxel_to_mm(per_axis_err_vox, spacing)
        hm_analysis = analyze_heatmap(heatmap, gt_norm, loc_input_size)

        side = case_id.rsplit("_", 1)[-1] if "_" in case_id else "unknown"

        result = {
            "idx": idx, "case_id": case_id, "side": side,
            "nii_path": case_info["nii_path"],
            "spacing": spacing,
            "gt_norm": gt_norm, "pred_norm": pred_norm,
            "gt_vox": gt_vox, "pred_vox": pred_vox,
            "wb_shape": wb_shape,
            "error_vox": error_vox, "error_mm": error_mm,
            "per_axis_err_vox": per_axis_err_vox,
            "per_axis_err_mm": per_axis_err_mm,
            "heatmap": heatmap,
            **hm_analysis,
        }
        all_results.append(result)

        print(f"  [{len(all_results)}/{len(target_indices)}] {case_id} ({side}): "
              f"{error_vox:.1f} vox / {error_mm:.2f} mm  |  "
              f"peak={hm_analysis['peak_value']:.2f}  "
              f"sharp={hm_analysis['sharpness']:.1f}")

    # --- 单例可视化 ---
    if not args.no_vis:
        # eval_all 模式: 只画 best 5 + worst 5
        if args.eval_all and len(all_results) > 10:
            sorted_results = sorted(all_results, key=lambda r: r["error_mm"])
            vis_results = sorted_results[:5] + sorted_results[-5:]
            print(f"\n可视化 best 5 + worst 5 cases...")
        else:
            vis_results = all_results

        for r in vis_results:
            nii = nib.load(r["nii_path"])
            wb_raw = nii.get_fdata().astype(np.float32)
            if wb_raw.ndim == 4:
                wb_raw = wb_raw[:, :, :, 0]
            visualize_single_case(
                wb_raw, r["gt_vox"], r["pred_vox"], r["heatmap"],
                r["case_id"], args.output_dir, loc_input_size,
                r["error_vox"], r["error_mm"], r["spacing"],
                {
                    "peak_value": r["peak_value"],
                    "sharpness": r["sharpness"],
                    "peak_gt_dist_vox": r["peak_gt_dist_vox"],
                },
            )

    # --- 综合仪表板 + 排名图 ---
    if len(all_results) > 1:
        print(f"\n生成综合仪表板...")
        plot_comprehensive_dashboard(all_results, args.output_dir, ckpt_name)
        plot_per_case_ranking(all_results, args.output_dir)

    # --- 详细报告 ---
    print_detailed_report(all_results)

    # --- CSV 导出 ---
    export_csv(all_results, args.output_dir)

    # --- NIfTI 导出 ---
    if args.save_nifti:
        print(f"\n保存 heatmap NIfTI...")
        save_heatmap_nifti(all_results, val_dataset, args.output_dir, loc_input_size)

    print(f"\n完成! 所有结果保存在: {args.output_dir}/")
    print(f"  - 单例可视化: {args.output_dir}/<case_id>_verify.png")
    print(f"  - 综合仪表板: {args.output_dir}/verification_dashboard.png")
    print(f"  - 排名图:     {args.output_dir}/error_ranking_mm.png")
    print(f"  - CSV 结果:   {args.output_dir}/verification_results.csv")
    if args.save_nifti:
        print(f"  - NIfTI:      {args.output_dir}/heatmap_nifti/")


if __name__ == "__main__":
    main()
