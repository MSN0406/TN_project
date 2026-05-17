"""
visualize_prediction.py
=======================
LocSegNet Phase 1 定位结果可视化 & 验证.

功能:
  1. 单例可视化: MRI 三视图 + Heatmap 叠加 + Centroid 局部放大
  2. 全验证集评估: 误差分布、ipsi/contra 对比、per-case 排名
  3. 训练曲线: 从 CSV 日志读取 loss 和 val_dist 画图
  4. 多 checkpoint 对比: 对比不同 epoch 的精度

用法:
    # 1) 快速可视化验证集前 5 个 case
    python visualize_prediction.py --checkpoint checkpoints/phase1_epoch70.pth

    # 2) 全验证集评估 + 误差统计
    python visualize_prediction.py --checkpoint checkpoints/phase1_epoch70.pth --eval_all

    # 3) 指定 case 可视化
    python visualize_prediction.py --checkpoint checkpoints/phase1_epoch70.pth --case_id 01612716_ipsi

    # 4) 画训练曲线
    python visualize_prediction.py --plot_training --csv logs/phase1_metrics.csv

    # 5) 多 checkpoint 对比
    python visualize_prediction.py --compare checkpoints/phase1_epoch50.pth checkpoints/phase1_epoch70.pth

    # 6) 指定输出目录和 GPU
    python visualize_prediction.py --checkpoint checkpoints/phase1_epoch70.pth --output_dir vis_results --gpu_id 2
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.locseg_net import LocSegNet
from data.dataset import TNLocSegDataset, get_train_val_split

# ----------------------------------------------------------------
# 自定义 colormap: 透明 → 红 → 黄 (用于 heatmap overlay)
# ----------------------------------------------------------------
HEATMAP_COLORS = [
    (0.0, 0.0, 0.0, 0.0),   # 透明
    (0.8, 0.0, 0.0, 0.4),   # 暗红半透明
    (1.0, 0.2, 0.0, 0.6),   # 红
    (1.0, 0.6, 0.0, 0.8),   # 橙
    (1.0, 1.0, 0.0, 0.95),  # 黄
]
HEATMAP_CMAP = LinearSegmentedColormap.from_list("heatmap_overlay", HEATMAP_COLORS, N=256)


# ================================================================
# 模型加载 & 推理
# ================================================================

def load_model(cfg, checkpoint_path, device):
    """加载模型和 checkpoint."""
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

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    phase = ckpt.get("phase", "?")
    print(f"[OK] 模型加载: {checkpoint_path} (phase={phase}, epoch={epoch})")
    return model, ckpt


def predict_single(model, wb_tensor, device):
    """对单个 whole brain 做定位预测."""
    model.eval()
    with torch.no_grad():
        wb = wb_tensor.unsqueeze(0).to(device)  # (1, 1, D, H, W)
        heatmap, centroid_norm = model.forward_loc(wb)
    return {
        "centroid_norm": centroid_norm.cpu().squeeze(0).numpy(),   # (3,)
        "heatmap": heatmap.cpu().squeeze(0).squeeze(0).numpy(),    # (D, H, W)
    }


def run_eval(model, val_dataset, device, loc_input_size):
    """在验证集上运行推理, 返回所有结果."""
    results = []
    for idx in range(len(val_dataset)):
        sample = val_dataset[idx]
        case_id = sample["case_id"]
        wb_tensor = sample["whole_brain"]
        gt_norm = sample["centroid_norm"].numpy()
        wb_shape = sample["wb_shape"].numpy()

        pred = predict_single(model, wb_tensor, device)
        pred_norm = pred["centroid_norm"]

        gt_vox = gt_norm * (wb_shape - 1)
        pred_vox = pred_norm * (wb_shape - 1)
        error = np.linalg.norm(gt_vox - pred_vox)
        per_axis_err = np.abs(gt_vox - pred_vox)

        # 解析 side (ipsi/contra)
        side = case_id.rsplit("_", 1)[-1] if "_" in case_id else "unknown"

        results.append({
            "idx": idx,
            "case_id": case_id,
            "side": side,
            "gt_norm": gt_norm,
            "pred_norm": pred_norm,
            "gt_vox": gt_vox,
            "pred_vox": pred_vox,
            "error": error,
            "per_axis_err": per_axis_err,
            "heatmap": pred["heatmap"],
            "wb_shape": wb_shape,
        })
        print(f"  [{idx+1}/{len(val_dataset)}] {case_id} ({side}): error = {error:.1f} voxels")

    return results


# ================================================================
# 可视化函数
# ================================================================

def _normalize_image(img):
    """归一化到 [0, 1] 用于显示."""
    vmin, vmax = np.percentile(img[img > 0], [1, 99]) if (img > 0).any() else (img.min(), img.max())
    img_n = np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return img_n


def visualize_case(wb_raw, gt_vox, pred_vox, heatmap, case_id, output_dir,
                   loc_input_size, error):
    """
    单个 case 的综合可视化 (3 行 3 列):
      行1: MRI 三视图 + GT/Pred centroid 标注
      行2: MRI + Heatmap 叠加 (alpha blend)
      行3: Centroid 局部放大 (zoom in ±30 voxels)
    """
    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(3, 3, figure=fig, hspace=0.25, wspace=0.15)

    fig.suptitle(
        f"Case: {case_id}    |    "
        f"GT: [{gt_vox[0]:.0f}, {gt_vox[1]:.0f}, {gt_vox[2]:.0f}]    "
        f"Pred: [{pred_vox[0]:.1f}, {pred_vox[1]:.1f}, {pred_vox[2]:.1f}]    "
        f"Error: {error:.1f} voxels",
        fontsize=14, fontweight="bold", y=0.98,
    )

    wb = _normalize_image(wb_raw)
    gt = gt_vox.astype(int)
    pred = pred_vox

    view_names = ["Sagittal (dim0)", "Coronal (dim1)", "Axial (dim2)"]

    # 三视图切片 (在 GT centroid 所在层切)
    slices_wb = [
        wb[np.clip(gt[0], 0, wb.shape[0]-1), :, :],
        wb[:, np.clip(gt[1], 0, wb.shape[1]-1), :],
        wb[:, :, np.clip(gt[2], 0, wb.shape[2]-1)],
    ]
    # 每个视图中 GT/Pred 的 2D 坐标 (col, row)
    markers = [
        ((gt[2], gt[1]), (pred[2], pred[1])),   # dim0 slice: axes are dim1(row), dim2(col)
        ((gt[2], gt[0]), (pred[2], pred[0])),   # dim1 slice: axes are dim0(row), dim2(col)
        ((gt[1], gt[0]), (pred[1], pred[0])),   # dim2 slice: axes are dim0(row), dim1(col)
    ]

    # ---- 行1: MRI 三视图 + centroid ----
    for col in range(3):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(slices_wb[col], cmap="gray", origin="lower", aspect="auto")
        gt_xy, pred_xy = markers[col]
        ax.plot(*gt_xy, "r+", markersize=18, markeredgewidth=2.5, label="GT")
        ax.plot(*pred_xy, "c*", markersize=14, markeredgewidth=1.5, label="Pred")
        # 画连接线
        ax.plot([gt_xy[0], pred_xy[0]], [gt_xy[1], pred_xy[1]],
                "y--", linewidth=1.2, alpha=0.7)
        ax.set_title(f"{view_names[col]} (slice={gt[col]})", fontsize=11)
        if col == 0:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
        ax.axis("off")

    # ---- 行2: Heatmap 叠加到 MRI ----
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    wb_shape = np.array(wb_raw.shape[:3], dtype=np.float32)
    hm_shape = np.array(loc_input_size, dtype=np.float32)
    gt_hm = (gt_vox / (wb_shape - 1) * (hm_shape - 1)).astype(int)
    pred_hm = pred_vox / (wb_shape - 1) * (hm_shape - 1)

    # 将 whole brain 降采样到 heatmap 尺寸做 overlay
    from scipy.ndimage import zoom as scipy_zoom
    scale = hm_shape / wb_shape
    wb_low = scipy_zoom(wb, scale, order=1)

    slices_hm_wb = [
        wb_low[np.clip(gt_hm[0], 0, wb_low.shape[0]-1), :, :],
        wb_low[:, np.clip(gt_hm[1], 0, wb_low.shape[1]-1), :],
        wb_low[:, :, np.clip(gt_hm[2], 0, wb_low.shape[2]-1)],
    ]
    slices_hm = [
        heatmap_norm[np.clip(gt_hm[0], 0, heatmap_norm.shape[0]-1), :, :],
        heatmap_norm[:, np.clip(gt_hm[1], 0, heatmap_norm.shape[1]-1), :],
        heatmap_norm[:, :, np.clip(gt_hm[2], 0, heatmap_norm.shape[2]-1)],
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

    # ---- 行3: Centroid 局部放大 (±30 voxels) ----
    zoom_radius = 30
    for col in range(3):
        ax = fig.add_subplot(gs[2, col])
        sl = slices_wb[col]
        gt_xy, pred_xy = markers[col]
        # 计算 zoom 区域
        cx, cy = int(gt_xy[0]), int(gt_xy[1])
        x0 = max(0, cx - zoom_radius)
        x1 = min(sl.shape[1], cx + zoom_radius)
        y0 = max(0, cy - zoom_radius)
        y1 = min(sl.shape[0], cy + zoom_radius)
        crop = sl[y0:y1, x0:x1]

        ax.imshow(crop, cmap="gray", origin="lower", aspect="auto",
                  extent=[x0, x1, y0, y1])
        ax.plot(*gt_xy, "r+", markersize=22, markeredgewidth=3, label="GT")
        ax.plot(*pred_xy, "c*", markersize=18, markeredgewidth=2, label="Pred")
        ax.plot([gt_xy[0], pred_xy[0]], [gt_xy[1], pred_xy[1]],
                "y--", linewidth=1.5, alpha=0.8)
        # 画 crop box 框
        from matplotlib.patches import Circle
        circle_gt = Circle(gt_xy, radius=3, fill=False, edgecolor="red",
                           linewidth=1.5, linestyle="--")
        circle_pred = Circle(pred_xy, radius=3, fill=False, edgecolor="cyan",
                             linewidth=1.5, linestyle="--")
        ax.add_patch(circle_gt)
        ax.add_patch(circle_pred)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(f"Zoom-in ({view_names[col]})", fontsize=11)
        if col == 0:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
        ax.axis("off")

    save_path = os.path.join(output_dir, f"{case_id}_localization.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    -> 保存: {save_path}")
    return save_path


def visualize_error_summary(results, output_dir, ckpt_name=""):
    """
    验证集误差汇总图 (2x2):
      (0,0) 误差直方图
      (0,1) 误差 CDF 曲线
      (1,0) Ipsi vs Contra 箱线图
      (1,1) Per-axis 误差对比
    """
    errors = np.array([r["error"] for r in results])
    case_ids = [r["case_id"] for r in results]
    sides = [r["side"] for r in results]
    per_axis = np.array([r["per_axis_err"] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Validation Error Summary  ({len(errors)} cases)  {ckpt_name}",
                 fontsize=15, fontweight="bold")

    # ---- (0,0) 误差直方图 ----
    ax = axes[0, 0]
    ax.hist(errors, bins=25, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(errors), color="red", linestyle="--", lw=2,
               label=f"Mean: {np.mean(errors):.1f}")
    ax.axvline(np.median(errors), color="orange", linestyle="--", lw=2,
               label=f"Median: {np.median(errors):.1f}")
    ax.set_xlabel("Localization Error (voxels)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Error Distribution", fontsize=13)
    ax.legend(fontsize=10)

    # ---- (0,1) CDF 曲线 ----
    ax = axes[0, 1]
    sorted_err = np.sort(errors)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    ax.plot(sorted_err, cdf, "b-", linewidth=2)
    ax.fill_between(sorted_err, 0, cdf, alpha=0.15, color="blue")
    # 标记关键阈值
    for thr, color in [(10, "green"), (20, "orange"), (30, "red")]:
        pct = (errors < thr).mean() * 100
        ax.axvline(thr, color=color, linestyle=":", lw=1.5, alpha=0.7)
        ax.annotate(f"<{thr}: {pct:.0f}%", xy=(thr, pct), fontsize=9,
                    xytext=(thr + 1, pct - 8), color=color, fontweight="bold")
    ax.set_xlabel("Error Threshold (voxels)", fontsize=12)
    ax.set_ylabel("Cumulative %", fontsize=12)
    ax.set_title("Cumulative Distribution", fontsize=13)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # ---- (1,0) Ipsi vs Contra 箱线图 ----
    ax = axes[1, 0]
    ipsi_err = [r["error"] for r in results if r["side"] == "ipsi"]
    contra_err = [r["error"] for r in results if r["side"] == "contra"]
    data_box = []
    labels_box = []
    if ipsi_err:
        data_box.append(ipsi_err)
        labels_box.append(f"Ipsi (n={len(ipsi_err)})\nMean={np.mean(ipsi_err):.1f}")
    if contra_err:
        data_box.append(contra_err)
        labels_box.append(f"Contra (n={len(contra_err)})\nMean={np.mean(contra_err):.1f}")
    if data_box:
        bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True, widths=0.5)
        colors_box = ["#3498db", "#e74c3c"]
        for patch, color in zip(bp["boxes"], colors_box[:len(data_box)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
    ax.set_ylabel("Error (voxels)", fontsize=12)
    ax.set_title("Ipsi vs Contra", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    # ---- (1,1) Per-axis 误差 ----
    ax = axes[1, 1]
    axis_names = ["Dim0 (L-R)", "Dim1 (P-A)", "Dim2 (I-S)"]
    means = per_axis.mean(axis=0)
    stds = per_axis.std(axis=0)
    x = np.arange(3)
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=["#2ecc71", "#3498db", "#e67e22"],
                  alpha=0.8, edgecolor="white", linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(axis_names, fontsize=11)
    ax.set_ylabel("Absolute Error (voxels)", fontsize=12)
    ax.set_title("Per-axis Error", fontsize=13)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{m:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "error_summary.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    -> 保存: {save_path}")

    # ---- Per-case 排名图 (单独保存) ----
    fig2, ax2 = plt.subplots(figsize=(10, max(6, len(errors) * 0.22)))
    sorted_idx = np.argsort(errors)[::-1]
    sorted_errors = errors[sorted_idx]
    sorted_ids = [case_ids[i] for i in sorted_idx]
    sorted_sides = [sides[i] for i in sorted_idx]

    n_show = min(40, len(sorted_errors))
    colors_bar = []
    for e, s in zip(sorted_errors[:n_show], sorted_sides[:n_show]):
        if e > 50:
            colors_bar.append("#e74c3c")
        elif e > 30:
            colors_bar.append("#f39c12")
        else:
            colors_bar.append("#2ecc71")
    ax2.barh(range(n_show), sorted_errors[:n_show], color=colors_bar, edgecolor="white")
    ax2.set_yticks(range(n_show))
    labels_ranked = [f"{sorted_ids[i]} ({sorted_sides[i]})" for i in range(n_show)]
    ax2.set_yticklabels(labels_ranked, fontsize=7)
    ax2.set_xlabel("Error (voxels)", fontsize=12)
    ax2.set_title(f"Per-case Error Ranking (showing top {n_show})", fontsize=13)
    ax2.invert_yaxis()
    legend_patches = [
        mpatches.Patch(color="#e74c3c", label="> 50 vox"),
        mpatches.Patch(color="#f39c12", label="30-50 vox"),
        mpatches.Patch(color="#2ecc71", label="< 30 vox"),
    ]
    ax2.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    save_path2 = os.path.join(output_dir, "error_ranking.png")
    fig2.savefig(save_path2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"    -> 保存: {save_path2}")

    # ---- 打印统计 ----
    print(f"\n{'='*55}")
    print(f"  验证集定位误差统计 ({len(errors)} cases)")
    print(f"{'='*55}")
    print(f"  Mean:    {np.mean(errors):>7.2f} voxels")
    print(f"  Median:  {np.median(errors):>7.2f} voxels")
    print(f"  Std:     {np.std(errors):>7.2f} voxels")
    print(f"  Min:     {np.min(errors):>7.2f} voxels")
    print(f"  Max:     {np.max(errors):>7.2f} voxels")
    print(f"  <10 vox: {(errors<10).sum():>3}/{len(errors)} ({(errors<10).mean()*100:.1f}%)")
    print(f"  <20 vox: {(errors<20).sum():>3}/{len(errors)} ({(errors<20).mean()*100:.1f}%)")
    print(f"  <30 vox: {(errors<30).sum():>3}/{len(errors)} ({(errors<30).mean()*100:.1f}%)")
    if ipsi_err:
        print(f"  Ipsi  mean: {np.mean(ipsi_err):.2f}, median: {np.median(ipsi_err):.2f}")
    if contra_err:
        print(f"  Contra mean: {np.mean(contra_err):.2f}, median: {np.median(contra_err):.2f}")
    print(f"  Per-axis mean: dim0={means[0]:.2f}, dim1={means[1]:.2f}, dim2={means[2]:.2f}")
    print(f"{'='*55}")

    return save_path


def plot_training_curves(csv_path, output_dir):
    """从 CSV 日志画训练曲线."""
    epochs, losses, val_dists, lrs = [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["train_loss"]))
            val_dists.append(float(row["val_dist_voxels"]) if row["val_dist_voxels"] else None)
            lrs.append(float(row["lr"]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training Curves ({os.path.basename(csv_path)})", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, losses, "b-", linewidth=1.2, alpha=0.7, label="Train Loss")
    # 平滑曲线
    if len(losses) > 5:
        window = min(5, len(losses))
        smooth = np.convolve(losses, np.ones(window)/window, mode="valid")
        ax.plot(epochs[window-1:], smooth, "r-", linewidth=2, label=f"Smoothed (w={window})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Val Dist
    ax = axes[1]
    val_epochs = [e for e, v in zip(epochs, val_dists) if v is not None]
    val_vals = [v for v in val_dists if v is not None]
    ax.plot(val_epochs, val_vals, "go-", linewidth=2, markersize=5, label="Val Dist")
    if val_vals:
        best_idx = np.argmin(val_vals)
        ax.plot(val_epochs[best_idx], val_vals[best_idx], "r*", markersize=15,
                label=f"Best: {val_vals[best_idx]:.2f} @ ep{val_epochs[best_idx]}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Distance (voxels)")
    ax.set_title("Validation Localization Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[2]
    ax.plot(epochs, lrs, "m-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    -> 保存: {save_path}")
    return save_path


def compare_checkpoints(cfg, ckpt_paths, val_dataset, device, output_dir, loc_input_size):
    """多 checkpoint 对比."""
    all_ckpt_results = {}
    for ckpt_path in ckpt_paths:
        name = os.path.basename(ckpt_path).replace(".pth", "")
        print(f"\n--- 评估: {name} ---")
        model, _ = load_model(cfg, ckpt_path, device)
        results = run_eval(model, val_dataset, device, loc_input_size)
        all_ckpt_results[name] = results
        del model
        torch.cuda.empty_cache()

    # 画对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Checkpoint Comparison", fontsize=14, fontweight="bold")

    # 箱线图对比
    ax = axes[0]
    data = []
    labels = []
    for name, results in all_ckpt_results.items():
        errors = [r["error"] for r in results]
        data.append(errors)
        labels.append(f"{name}\n(mean={np.mean(errors):.1f})")
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    colors_bp = plt.cm.Set2(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Error (voxels)")
    ax.set_title("Error Distribution by Checkpoint")
    ax.grid(True, alpha=0.3, axis="y")

    # CDF 对比
    ax = axes[1]
    for i, (name, results) in enumerate(all_ckpt_results.items()):
        errors = np.sort([r["error"] for r in results])
        cdf = np.arange(1, len(errors) + 1) / len(errors) * 100
        ax.plot(errors, cdf, linewidth=2, label=name)
    ax.set_xlabel("Error (voxels)")
    ax.set_ylabel("Cumulative %")
    ax.set_title("CDF Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "checkpoint_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    -> 保存: {save_path}")


# ================================================================
# 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LocSegNet 定位可视化 & 验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint 路径")
    parser.add_argument("--output_dir", type=str, default="vis_results",
                        help="输出目录 (默认 vis_results/)")
    parser.add_argument("--num_cases", type=int, default=5,
                        help="可视化 case 数量")
    parser.add_argument("--case_id", type=str, default=None,
                        help="指定 case id")
    parser.add_argument("--gpu_id", type=int, default=None,
                        help="GPU 编号")
    parser.add_argument("--eval_all", action="store_true",
                        help="全验证集评估 + 误差统计")
    parser.add_argument("--plot_training", action="store_true",
                        help="画训练曲线")
    parser.add_argument("--csv", type=str, default="logs/phase1_metrics.csv",
                        help="训练 CSV 日志路径 (配合 --plot_training)")
    parser.add_argument("--compare", nargs="+", type=str, default=None,
                        help="多 checkpoint 对比 (空格分隔多个路径)")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 设备
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

    # ---- 模式 1: 画训练曲线 ----
    if args.plot_training:
        print("\n[训练曲线]")
        plot_training_curves(args.csv, args.output_dir)
        if not args.checkpoint and not args.compare:
            print(f"\n完成! 结果: {args.output_dir}/")
            return

    # ---- 准备验证集 ----
    if args.checkpoint or args.compare:
        prepared_dir = cfg["data"]["prepared_dir"]
        _, val_ids = get_train_val_split(
            prepared_dir,
            train_ratio=cfg["data"]["train_val_split"],
            seed=cfg["data"]["random_seed"],
        )
        val_dataset = TNLocSegDataset(prepared_dir, case_ids=val_ids, phase=1)
        print(f"验证集: {len(val_dataset)} cases")

    # ---- 模式 2: 多 checkpoint 对比 ----
    if args.compare:
        compare_checkpoints(cfg, args.compare, val_dataset, device,
                            args.output_dir, loc_input_size)
        print(f"\n完成! 结果: {args.output_dir}/")
        return

    # ---- 模式 3: 单 checkpoint 可视化 + 评估 ----
    if args.checkpoint:
        model, ckpt = load_model(cfg, args.checkpoint, device)
        ckpt_name = os.path.basename(args.checkpoint)

        # 确定 case 列表
        if args.case_id:
            target_indices = [i for i, c in enumerate(val_dataset.cases)
                              if c["case_id"] == args.case_id]
            if not target_indices:
                print(f"[错误] '{args.case_id}' 不在验证集中")
                print(f"可选: {[c['case_id'] for c in val_dataset.cases[:20]]}")
                return
        elif args.eval_all:
            target_indices = list(range(len(val_dataset)))
        else:
            target_indices = list(range(min(args.num_cases, len(val_dataset))))

        # 推理
        print(f"\n开始推理 ({len(target_indices)} cases)...")
        all_results = []
        for idx in target_indices:
            sample = val_dataset[idx]
            case_id = sample["case_id"]
            wb_tensor = sample["whole_brain"]
            gt_norm = sample["centroid_norm"].numpy()
            wb_shape = sample["wb_shape"].numpy()
            side = case_id.rsplit("_", 1)[-1] if "_" in case_id else "unknown"

            pred = predict_single(model, wb_tensor, device)
            pred_norm = pred["centroid_norm"]

            gt_vox = gt_norm * (wb_shape - 1)
            pred_vox = pred_norm * (wb_shape - 1)
            error = np.linalg.norm(gt_vox - pred_vox)
            per_axis_err = np.abs(gt_vox - pred_vox)

            all_results.append({
                "idx": idx, "case_id": case_id, "side": side,
                "gt_norm": gt_norm, "pred_norm": pred_norm,
                "gt_vox": gt_vox, "pred_vox": pred_vox,
                "error": error, "per_axis_err": per_axis_err,
                "heatmap": pred["heatmap"], "wb_shape": wb_shape,
            })
            print(f"  [{len(all_results)}/{len(target_indices)}] {case_id} ({side}): "
                  f"error = {error:.1f} voxels")

            # 生成可视化 (eval_all 时最多画 best 5 + worst 5)
            should_visualize = not args.eval_all or len(target_indices) <= 10
            if should_visualize:
                case_info = val_dataset.cases[idx]
                nii = nib.load(case_info["nii_path"])
                wb_raw = nii.get_fdata().astype(np.float32)
                if wb_raw.ndim == 4:
                    wb_raw = wb_raw[:, :, :, 0]
                visualize_case(wb_raw, gt_vox, pred_vox, pred["heatmap"],
                               case_id, args.output_dir, loc_input_size, error)

        # eval_all 模式: 额外画 best 5 + worst 5
        if args.eval_all and len(target_indices) > 10:
            sorted_results = sorted(all_results, key=lambda r: r["error"])
            special_cases = sorted_results[:5] + sorted_results[-5:]
            print(f"\n可视化 best 5 + worst 5 cases...")
            for r in special_cases:
                case_info = val_dataset.cases[r["idx"]]
                nii = nib.load(case_info["nii_path"])
                wb_raw = nii.get_fdata().astype(np.float32)
                if wb_raw.ndim == 4:
                    wb_raw = wb_raw[:, :, :, 0]
                tag = "BEST" if r["error"] == sorted_results[0]["error"] else \
                      "WORST" if r["error"] == sorted_results[-1]["error"] else ""
                visualize_case(wb_raw, r["gt_vox"], r["pred_vox"], r["heatmap"],
                               r["case_id"], args.output_dir, loc_input_size, r["error"])

        # 误差汇总图
        if len(all_results) > 1:
            visualize_error_summary(all_results, args.output_dir, ckpt_name)

    print(f"\n完成! 所有结果保存在: {args.output_dir}/")


if __name__ == "__main__":
    main()
