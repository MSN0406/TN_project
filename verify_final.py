"""
verify_final.py
===============
LocSegNet 三阶段模型最终验证 & 可视化.

评估内容:
  1. 定位精度: 双头预测 centroid vs GT (体素 + mm)
  2. 分割质量: per-class Dice (nerve, vessel)
  3. 综合可视化: 定位 + 分割叠加

用法:
    python verify_final.py --config configs/default.yaml \
        --checkpoint runs/run_xxx/checkpoints/phase3_final.pth \
        --output_dir verify_final_results --gpu 1
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from models.locseg_net import LocSegNet
from models.loc_encoder import soft_argmax_3d
from data.dataset import TNLocSegDataset, get_train_val_split


# ================================================================
# 工具函数
# ================================================================

def get_voxel_spacing(nii_path):
    """从 NIfTI header 获取体素 spacing."""
    nii = nib.load(nii_path)
    return np.abs(nii.header.get_zooms()[:3]).astype(np.float64)


def compute_dice_2d(pred, gt, num_classes=3, orig_size=48):
    """
    2D slice-wise Dice score.

    1. Center-crop pred 和 gt 到 orig_size³ (去掉嵌入 padding)
    2. 沿 3 个轴分别逐层计算 2D Dice
    3. 只统计 GT 有前景的层, 取平均
    4. 3 个轴结果再取平均

    参数:
        pred: (D, H, W) int numpy 预测
        gt:   (D, H, W) int numpy 标注
        num_classes: 类别数
        orig_size: 原始标注尺寸 (48)

    返回:
        dice: {class_idx: mean_2d_dice}
    """
    # Center-crop 到原始标注区域
    pred_crop = center_crop_3d(pred, (orig_size, orig_size, orig_size))
    gt_crop = center_crop_3d(gt, (orig_size, orig_size, orig_size))

    fg_classes = list(range(1, num_classes))  # [1, 2]

    dice = {}
    for c in fg_classes:
        axis_dices = []
        for axis in range(3):  # dim0, dim1, dim2
            slice_dices = []
            n_slices = pred_crop.shape[axis]
            for s in range(n_slices):
                if axis == 0:
                    g_sl = gt_crop[s, :, :]
                    p_sl = pred_crop[s, :, :]
                elif axis == 1:
                    g_sl = gt_crop[:, s, :]
                    p_sl = pred_crop[:, s, :]
                else:
                    g_sl = gt_crop[:, :, s]
                    p_sl = pred_crop[:, :, s]

                # 只统计 GT 中所有前景类都出现的层
                all_present = all((g_sl == fc).any() for fc in fg_classes)
                if not all_present:
                    continue

                gt_fg = (g_sl == c).astype(float)
                pred_fg = (p_sl == c).astype(float)
                intersection = (pred_fg * gt_fg).sum()
                union = pred_fg.sum() + gt_fg.sum()
                if union > 0:
                    slice_dices.append(2 * intersection / union)

            if slice_dices:
                axis_dices.append(np.mean(slice_dices))

        if axis_dices:
            dice[c] = np.mean(axis_dices)
        else:
            dice[c] = float('nan')

    return dice


def center_crop_3d(arr, target_shape):
    """Center-crop 3D array/tensor 到 target 尺寸."""
    d, h, w = arr.shape[-3:]
    td, th, tw = target_shape
    if d == td and h == th and w == tw:
        return arr
    d0 = (d - td) // 2
    h0 = (h - th) // 2
    w0 = (w - tw) // 2
    if arr.ndim == 3:
        return arr[d0:d0+td, h0:h0+th, w0:w0+tw]
    elif arr.ndim == 5:  # (B, C, D, H, W)
        return arr[:, :, d0:d0+td, h0:h0+th, w0:w0+tw]
    else:
        raise ValueError(f"Unsupported ndim={arr.ndim}")


# ================================================================
# 模型加载 & 推理
# ================================================================

def load_model(cfg, checkpoint_path, device):
    """加载模型 (支持 nnunet/custom seg_arch + dual_seg + dropout)."""
    dual_seg = cfg["segmentation"].get("dual_seg", False)
    dropout = cfg["segmentation"].get("dropout", 0.0)
    seg_arch = cfg["segmentation"].get("seg_arch", "nnunet")

    model = LocSegNet(
        loc_input_size=tuple(cfg["localization"]["input_size"]),
        seg_crop_size=tuple(cfg["segmentation"]["crop_size"]),
        num_classes=cfg["segmentation"]["num_classes"],
        loc_channels=tuple(cfg["localization"]["encoder_channels"]),
        seg_channels=tuple(cfg["segmentation"].get("encoder_channels", [32, 64, 128, 256, 512])),
        deep_supervision=cfg["segmentation"]["deep_supervision"],
        dual_seg=dual_seg,
        dropout=dropout,
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
    print(f"[OK] 模型加载: {checkpoint_path} (phase={phase}, epoch={epoch}, seg_arch={seg_arch})")
    return model


def predict_single(model, wb_tensor, device, gt_centroid_norm, mask_shape):
    """
    对单个 case 做完整推理 (定位 + 分割).

    参数:
        mask_shape: GT mask 的空间尺寸 (D, H, W), 用于 center-crop 分割输出

    返回:
        dict 包含定位和分割预测结果
    """
    model.eval()
    with torch.no_grad():
        wb = wb_tensor.unsqueeze(0).to(device)
        gt = torch.from_numpy(gt_centroid_norm).unsqueeze(0).float().to(device)

        # 定位 (dual-head, 用 GT 选择正确的 head)
        loc_result = model.forward_loc(wb, gt_centroid_norm=gt)
        pred_centroid = loc_result["centroid_norm"]

        # 分割 (用 GT centroid 裁剪, 公平评估分割质量)
        seg_out_raw, ds_out_raw, roi_gt = model.forward_seg(wb, gt)

        if model.dual_seg:
            # dual_seg: 两个 binary 网络输出合并
            nerve_out = seg_out_raw["nerve"]   # (1, 2, D, H, W)
            vessel_out = seg_out_raw["vessel"]
            # center-crop
            if nerve_out.shape[2] > mask_shape[0]:
                nerve_out = center_crop_3d(nerve_out, mask_shape)
                vessel_out = center_crop_3d(vessel_out, mask_shape)
            nerve_pred = torch.argmax(nerve_out, dim=1)   # (1, D, H, W)
            vessel_pred = torch.argmax(vessel_out, dim=1)
            # 合并
            combined = torch.zeros_like(nerve_pred)
            combined[nerve_pred == 1] = 1
            combined[vessel_pred == 1] = 2
            # 重叠: 置信度高的优先
            overlap = (nerve_pred == 1) & (vessel_pred == 1)
            if overlap.any():
                nc = torch.softmax(nerve_out, dim=1)[:, 1]
                vc = torch.softmax(vessel_out, dim=1)[:, 1]
                combined[overlap & (vc > nc)] = 2
                combined[overlap & (nc >= vc)] = 1
            seg_pred_gt = combined.squeeze(0).cpu().numpy()
        else:
            out_shape = seg_out_raw.shape[2:]
            if out_shape[0] > mask_shape[0]:
                seg_out_crop = center_crop_3d(seg_out_raw, mask_shape)
            else:
                seg_out_crop = seg_out_raw
            seg_pred_gt = torch.argmax(seg_out_crop, dim=1).squeeze(0).cpu().numpy()

        # ROI 图像
        roi_np_full = roi_gt.cpu().squeeze(0).squeeze(0).numpy()
        if roi_np_full.shape[0] > mask_shape[0]:
            roi_np_crop = center_crop_3d(roi_np_full, mask_shape)
        else:
            roi_np_crop = roi_np_full

    hm = loc_result["heatmap"].cpu().squeeze(0).squeeze(0).numpy()
    cn = pred_centroid.cpu().squeeze(0).numpy()

    return {
        "centroid_norm": cn,
        "heatmap": hm,
        "seg_pred_gt": seg_pred_gt,   # center-cropped, 与 mask 对齐
        "roi": roi_np_crop,            # center-cropped ROI 图像
    }


# ================================================================
# 可视化
# ================================================================

def _find_best_slice(gt_mask, axis, num_classes=3):
    """找到 GT 中所有前景类都出现且前景最多的层."""
    fg_classes = list(range(1, num_classes))
    best_s, best_count = gt_mask.shape[axis] // 2, 0  # 默认中心
    for s in range(gt_mask.shape[axis]):
        if axis == 0:
            sl = gt_mask[s, :, :]
        elif axis == 1:
            sl = gt_mask[:, s, :]
        else:
            sl = gt_mask[:, :, s]
        # 所有前景类都要出现
        if not all((sl == c).any() for c in fg_classes):
            continue
        count = sum((sl == c).sum() for c in fg_classes)
        if count > best_count:
            best_count = count
            best_s = s
    return best_s


def plot_seg_overlay(roi, gt_mask, pred_mask, case_id, save_path,
                     dice_scores, loc_error_mm, orig_size=48):
    """绘制分割结果叠加图 (在 48³ 区域, 选 nerve+vessel 都有的最佳层)."""
    # Center-crop 到 48³
    roi = center_crop_3d(roi, (orig_size, orig_size, orig_size))
    gt_mask = center_crop_3d(gt_mask, (orig_size, orig_size, orig_size))
    pred_mask = center_crop_3d(pred_mask, (orig_size, orig_size, orig_size))

    D, H, W = roi.shape
    # 每个轴找 nerve+vessel 都有的最佳层
    slices = [_find_best_slice(gt_mask, ax) for ax in range(3)]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(
        f"Case: {case_id}  (48³ region, best slice with both labels)\n"
        f"Loc Error: {loc_error_mm:.2f} mm | "
        f"Dice nerve: {dice_scores.get(1, float('nan')):.4f} | "
        f"Dice vessel: {dice_scores.get(2, float('nan')):.4f}",
        fontsize=14, fontweight="bold",
    )

    dim_names = ["Sagittal (dim0)", "Coronal (dim1)", "Axial (dim2)"]
    row_names = ["ROI Image", "GT Mask", "Pred Mask"]

    # 颜色映射: 0=透明, 1=red(nerve), 2=blue(vessel)
    colors = {1: [1, 0, 0], 2: [0, 0.5, 1]}

    for col, (dim, s) in enumerate(zip(range(3), slices)):
        # ROI 切片
        if dim == 0:
            roi_slice = roi[s, :, :]
            gt_slice = gt_mask[s, :, :]
            pred_slice = pred_mask[s, :, :]
        elif dim == 1:
            roi_slice = roi[:, s, :]
            gt_slice = gt_mask[:, s, :]
            pred_slice = pred_mask[:, s, :]
        else:
            roi_slice = roi[:, :, s]
            gt_slice = gt_mask[:, :, s]
            pred_slice = pred_mask[:, :, s]

        # Row 0: ROI image
        axes[0, col].imshow(roi_slice.T, cmap="gray", origin="lower")
        axes[0, col].set_title(f"{dim_names[col]} (slice={s})")
        axes[0, col].axis("off")

        # Row 1: GT overlay
        roi_rgb = np.stack([roi_slice.T] * 3, axis=-1)
        roi_rgb = (roi_rgb - roi_rgb.min()) / (roi_rgb.max() - roi_rgb.min() + 1e-8)
        gt_overlay = roi_rgb.copy()
        for c, color in colors.items():
            mask_c = (gt_slice.T == c)
            for ch in range(3):
                gt_overlay[:, :, ch][mask_c] = color[ch] * 0.7 + gt_overlay[:, :, ch][mask_c] * 0.3
        axes[1, col].imshow(gt_overlay, origin="lower")
        axes[1, col].set_title("GT Mask")
        axes[1, col].axis("off")

        # Row 2: Pred overlay
        pred_overlay = roi_rgb.copy()
        for c, color in colors.items():
            mask_c = (pred_slice.T == c)
            for ch in range(3):
                pred_overlay[:, :, ch][mask_c] = color[ch] * 0.7 + pred_overlay[:, :, ch][mask_c] * 0.3
        axes[2, col].imshow(pred_overlay, origin="lower")
        axes[2, col].set_title("Pred Mask")
        axes[2, col].axis("off")

    # 行标签
    for row, name in enumerate(row_names):
        axes[row, 0].set_ylabel(name, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary_dashboard(results, output_dir, ckpt_name):
    """绘制综合仪表板."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Final Model Verification | {len(results)} cases | {ckpt_name}\n"
        f"Loc Mean: {np.mean([r['error_mm'] for r in results]):.2f} mm | "
        f"Dice Nerve: {np.nanmean([r['dice'].get(1, float('nan')) for r in results]):.4f} | "
        f"Dice Vessel: {np.nanmean([r['dice'].get(2, float('nan')) for r in results]):.4f}",
        fontsize=14, fontweight="bold",
    )

    # --- 1. 定位误差分布 ---
    errors_mm = [r["error_mm"] for r in results]
    ax = axes[0, 0]
    ax.hist(errors_mm, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(errors_mm), color="red", linestyle="--", label=f"Mean: {np.mean(errors_mm):.2f}")
    ax.axvline(np.median(errors_mm), color="orange", linestyle="--", label=f"Median: {np.median(errors_mm):.2f}")
    ax.set_xlabel("Localization Error (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Localization Error Distribution")
    ax.legend()

    # --- 2. Per-axis 误差 ---
    ax = axes[0, 1]
    dim_names = ["Dim0\n(L-R)", "Dim1\n(P-A)", "Dim2\n(I-S)"]
    dim_errors = [
        [r["err_dim_mm"][0] for r in results],
        [r["err_dim_mm"][1] for r in results],
        [r["err_dim_mm"][2] for r in results],
    ]
    bp = ax.boxplot(dim_errors, tick_labels=dim_names, patch_artist=True)
    colors_bp = ["#4C72B0", "#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Error (mm)")
    ax.set_title("Per-axis Error")
    for i, d in enumerate(dim_errors):
        ax.text(i + 1, max(d) * 0.95, f"μ={np.mean(d):.1f}", ha="center", fontsize=9)

    # --- 3. Ipsi vs Contra ---
    ax = axes[0, 2]
    ipsi = [r["error_mm"] for r in results if r["side"] == "ipsi"]
    contra = [r["error_mm"] for r in results if r["side"] == "contra"]
    bp = ax.boxplot([ipsi, contra],
                     tick_labels=[f"Ipsi (n={len(ipsi)})\nMean={np.mean(ipsi):.2f}mm",
                                  f"Contra (n={len(contra)})\nMean={np.mean(contra):.2f}mm"],
                     patch_artist=True)
    bp["boxes"][0].set_facecolor("#4C72B0")
    bp["boxes"][1].set_facecolor("#C44E52")
    for b in bp["boxes"]:
        b.set_alpha(0.7)
    ax.set_ylabel("Error (mm)")
    ax.set_title("Ipsi vs Contra (Localization)")

    # --- 4. Dice 分布 ---
    ax = axes[1, 0]
    nerve_dice = [r["dice"].get(1, float('nan')) for r in results]
    vessel_dice = [r["dice"].get(2, float('nan')) for r in results]
    nerve_valid = [d for d in nerve_dice if not np.isnan(d)]
    vessel_valid = [d for d in vessel_dice if not np.isnan(d)]

    if nerve_valid:
        ax.hist(nerve_valid, bins=20, alpha=0.6, color="red", label=f"Nerve (n={len(nerve_valid)}, μ={np.mean(nerve_valid):.4f})")
    if vessel_valid:
        ax.hist(vessel_valid, bins=20, alpha=0.6, color="blue", label=f"Vessel (n={len(vessel_valid)}, μ={np.mean(vessel_valid):.4f})")
    ax.set_xlabel("Dice Score")
    ax.set_ylabel("Count")
    ax.set_title("2D Slice-wise Dice (48³ region, GT centroid)")
    ax.legend()

    # --- 5. Loc Error vs Dice ---
    ax = axes[1, 1]
    valid_results = [(r["error_mm"], r["dice"].get(1, float('nan'))) for r in results]
    valid_results = [(e, d) for e, d in valid_results if not np.isnan(d)]
    if valid_results:
        errs, dices = zip(*valid_results)
        sc = ax.scatter(errs, dices, c=errs, cmap="RdYlGn_r", s=30, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="Error (mm)")
    ax.set_xlabel("Localization Error (mm)")
    ax.set_ylabel("Nerve Dice")
    ax.set_title("Loc Error vs Seg Quality")

    # --- 6. CDF ---
    ax = axes[1, 2]
    sorted_err = np.sort(errors_mm)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    ax.plot(sorted_err, cdf, color="steelblue", linewidth=2)
    ax.fill_between(sorted_err, cdf, alpha=0.2, color="steelblue")
    for thresh in [5, 10]:
        pct = np.mean(np.array(errors_mm) < thresh) * 100
        ax.axvline(thresh, color="gray", linestyle=":", alpha=0.5)
        ax.text(thresh + 0.5, pct - 5, f"<{thresh}mm: {pct:.0f}%", fontsize=9, color="green")
    ax.set_xlabel("Error (mm)")
    ax.set_ylabel("Cumulative %")
    ax.set_title("CDF (Localization)")
    ax.set_ylim(0, 105)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "final_dashboard.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> 保存: {save_path}")


def plot_error_ranking(results, output_dir):
    """按误差排名的条形图."""
    sorted_results = sorted(results, key=lambda r: r["error_mm"], reverse=True)
    top_n = min(50, len(sorted_results))
    top = sorted_results[:top_n]

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.25)))
    names = [f"{r['case_id']} [{r['error_vox']:.0f}vox]" for r in top]
    errors = [r["error_mm"] for r in top]
    colors = ["#d32f2f" if e > 10 else "#ff9800" if e > 5 else "#4caf50" for e in errors]

    ax.barh(range(len(names)), errors, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Error (mm)")
    ax.set_title(f"Per-case Error Ranking (top {top_n}, mm)")

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d32f2f", label="> 10 mm"),
        Patch(facecolor="#ff9800", label="5-10 mm"),
        Patch(facecolor="#4caf50", label="< 5 mm"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "error_ranking_mm.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> 保存: {save_path}")


# ================================================================
# 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="LocSegNet 最终模型验证")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="verify_final_results")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--num_vis", type=int, default=10,
                        help="可视化 case 数量 (best 5 + worst 5)")
    parser.add_argument("--crop_size", type=int, default=None,
                        help="覆盖分割 crop_size (例如 48, 直接在 48³ 上推理)")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 设备
    device_cfg = cfg.get("device", {})
    gpu_id = args.gpu if args.gpu is not None else device_cfg.get("gpu_id", 0)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    num_classes = cfg["segmentation"]["num_classes"]
    class_names = cfg["segmentation"].get("class_names", ["bg", "nerve", "vessel"])
    crop_size = cfg["segmentation"]["crop_size"]

    # --- 加载模型 ---
    model = load_model(cfg, args.checkpoint, device)
    ckpt_name = os.path.basename(args.checkpoint)

    # 覆盖 crop_size (用于直接在 48³ 上推理)
    if args.crop_size is not None:
        cs = args.crop_size
        model.seg_crop_size = (cs, cs, cs)
        print(f"[Override] seg_crop_size = {model.seg_crop_size}")
    effective_crop = model.seg_crop_size

    # --- 验证集 ---
    _, val_ids = get_train_val_split(
        cfg["data"]["prepared_dir"],
        train_ratio=cfg["data"]["train_val_split"],
        seed=cfg["data"]["random_seed"],
    )
    val_dataset = TNLocSegDataset(
        cfg["data"]["prepared_dir"], case_ids=val_ids, phase=2, lr_flip_prob=0.0,
    )
    print(f"验证集: {len(val_dataset)} cases\n")

    # --- 推理 ---
    all_results = []
    print(f"开始推理 ({len(val_dataset)} cases)...")

    for idx in range(len(val_dataset)):
        sample = val_dataset[idx]
        case_id = sample["case_id"]
        wb_tensor = sample["whole_brain"]
        gt_norm = sample["centroid_norm"].numpy()
        wb_shape = sample["wb_shape"].numpy()

        # GT mask (64x64x64, 原始标注 48x48x48 居中嵌入)
        gt_mask = sample["mask"].numpy()  # (64, 64, 64)
        gt_mask[gt_mask >= num_classes] = 0
        gt_mask[gt_mask < 0] = 0

        # 如果 crop_size < mask 文件尺寸, center-crop mask 以匹配模型输出
        if effective_crop[0] < gt_mask.shape[0]:
            gt_mask = center_crop_3d(gt_mask, effective_crop)
        mask_shape = gt_mask.shape  # (crop_size³ or 64³)

        # spacing
        case_info = val_dataset.cases[idx]
        spacing = get_voxel_spacing(case_info["nii_path"])

        # 推理 (分割输出 center-crop 到 mask_shape)
        pred = predict_single(model, wb_tensor, device, gt_norm, mask_shape)
        pred_norm = pred["centroid_norm"]

        # 定位误差
        gt_vox = gt_norm * (wb_shape - 1)
        pred_vox = pred_norm * (wb_shape - 1)
        err_vox = np.abs(gt_vox - pred_vox)
        err_mm = err_vox * spacing
        error_vox = np.linalg.norm(err_vox)
        error_mm = np.linalg.norm(err_mm)

        # 分割 Dice: 2D slice-wise, 在原始 48³ 标注区域内计算
        seg_pred_gt = pred["seg_pred_gt"]  # (64, 64, 64) center-cropped
        orig_mask_size = cfg["segmentation"].get("mask_size", 64)
        # 原始标注是 48³, 嵌入到 mask_size³(64³)
        # info.json 中有 mask_shape_original, 这里用通用值
        orig_annot_size = 48
        dice = compute_dice_2d(seg_pred_gt, gt_mask, num_classes, orig_size=orig_annot_size)

        side = "ipsi" if "_ipsi" in case_id else "contra"

        result = {
            "case_id": case_id,
            "side": side,
            "error_vox": error_vox,
            "error_mm": error_mm,
            "err_dim_vox": err_vox,
            "err_dim_mm": err_mm,
            "dice": dice,
            "gt_vox": gt_vox,
            "pred_vox": pred_vox,
            "seg_pred_gt": seg_pred_gt,
            "gt_mask": gt_mask,
            "roi": pred["roi"],
            "spacing": spacing,
        }
        all_results.append(result)

        nerve_d = dice.get(1, float('nan'))
        vessel_d = dice.get(2, float('nan'))
        print(f"  [{idx+1}/{len(val_dataset)}] {case_id}: "
              f"{error_vox:.1f} vox / {error_mm:.2f} mm | "
              f"Dice nerve={nerve_d:.4f} vessel={vessel_d:.4f}")

    # --- 可视化 ---
    print(f"\n可视化 best {args.num_vis // 2} + worst {args.num_vis // 2} cases...")
    sorted_results = sorted(all_results, key=lambda r: r["error_mm"])
    n_half = args.num_vis // 2
    vis_cases = sorted_results[:n_half] + sorted_results[-n_half:]

    for r in vis_cases:
        save_path = os.path.join(args.output_dir, f"{r['case_id']}_seg.png")
        plot_seg_overlay(
            r["roi"], r["gt_mask"], r["seg_pred_gt"],
            r["case_id"], save_path, r["dice"], r["error_mm"],
        )
        print(f"    -> {save_path}")

    # --- 综合仪表板 ---
    print(f"\n生成综合仪表板...")
    plot_summary_dashboard(all_results, args.output_dir, ckpt_name)
    plot_error_ranking(all_results, args.output_dir)

    # --- 打印报告 ---
    errors_mm = [r["error_mm"] for r in all_results]
    errors_vox = [r["error_vox"] for r in all_results]
    nerve_dices = [r["dice"].get(1, float('nan')) for r in all_results]
    vessel_dices = [r["dice"].get(2, float('nan')) for r in all_results]
    nerve_valid = [d for d in nerve_dices if not np.isnan(d)]
    vessel_valid = [d for d in vessel_dices if not np.isnan(d)]

    print(f"\n{'='*60}")
    print(f"  最终模型验证报告  ({len(all_results)} cases)")
    print(f"{'='*60}")
    print(f"\n  === 定位精度 ===")
    print(f"  Mean:   {np.mean(errors_mm):.2f} mm / {np.mean(errors_vox):.2f} vox")
    print(f"  Median: {np.median(errors_mm):.2f} mm / {np.median(errors_vox):.2f} vox")
    print(f"  Std:    {np.std(errors_mm):.2f} mm")
    print(f"  Min:    {np.min(errors_mm):.2f} mm")
    print(f"  Max:    {np.max(errors_mm):.2f} mm")
    print(f"\n  < 5mm:  {np.mean(np.array(errors_mm) < 5) * 100:.1f}%")
    print(f"  <10mm:  {np.mean(np.array(errors_mm) < 10) * 100:.1f}%")
    print(f"  <15mm:  {np.mean(np.array(errors_mm) < 15) * 100:.1f}%")

    dim_names = ["Dim0 (L-R)", "Dim1 (P-A)", "Dim2 (I-S)"]
    print(f"\n  Per-axis (mm):")
    for i, dn in enumerate(dim_names):
        vals = [r["err_dim_mm"][i] for r in all_results]
        print(f"    {dn}: mean={np.mean(vals):.2f}, std={np.std(vals):.2f}, max={np.max(vals):.2f}")

    ipsi_err = [r["error_mm"] for r in all_results if r["side"] == "ipsi"]
    contra_err = [r["error_mm"] for r in all_results if r["side"] == "contra"]
    print(f"\n  Ipsi  (n={len(ipsi_err)}): mean={np.mean(ipsi_err):.2f} mm")
    print(f"  Contra(n={len(contra_err)}): mean={np.mean(contra_err):.2f} mm")

    print(f"\n  === 分割质量 (GT centroid) ===")
    if nerve_valid:
        print(f"  Nerve  Dice: mean={np.mean(nerve_valid):.4f}, "
              f"median={np.median(nerve_valid):.4f}, "
              f"std={np.std(nerve_valid):.4f} (n={len(nerve_valid)})")
    if vessel_valid:
        print(f"  Vessel Dice: mean={np.mean(vessel_valid):.4f}, "
              f"median={np.median(vessel_valid):.4f}, "
              f"std={np.std(vessel_valid):.4f} (n={len(vessel_valid)})")
    if nerve_valid and vessel_valid:
        mean_fg = (np.mean(nerve_valid) + np.mean(vessel_valid)) / 2
        print(f"  Mean FG Dice: {mean_fg:.4f}")
    print(f"{'='*60}")

    # --- CSV 导出 ---
    csv_path = os.path.join(args.output_dir, "final_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "side", "error_vox", "error_mm",
            "err_d0_mm", "err_d1_mm", "err_d2_mm",
            "dice_nerve", "dice_vessel",
        ])
        for r in all_results:
            writer.writerow([
                r["case_id"], r["side"],
                f"{r['error_vox']:.2f}", f"{r['error_mm']:.3f}",
                f"{r['err_dim_mm'][0]:.3f}", f"{r['err_dim_mm'][1]:.3f}", f"{r['err_dim_mm'][2]:.3f}",
                f"{r['dice'].get(1, float('nan')):.4f}",
                f"{r['dice'].get(2, float('nan')):.4f}",
            ])
    print(f"\n  CSV: {csv_path}")
    print(f"  完成! 所有结果保存在: {args.output_dir}/")


if __name__ == "__main__":
    main()
