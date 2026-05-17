"""
visualize_best_slice.py
=======================
逐 slice 计算 2D Dice, 找到 Dice 最高的 slice 进行可视化.

每个 case 一张图, 包含:
  - 上排: ROI 图像 | GT mask | Pred mask | GT vs Pred 对比
  - 下排: 该 slice 的 per-class Dice 信息, 以及 3 个轴的 Dice 曲线

用法:
    # 仅分割模式 (默认): 使用 GT centroid 裁剪 ROI, 仅运行 seg 网络
    python visualize_best_slice.py \
        --config configs/default.yaml \
        --mode seg \
        --checkpoint runs/run_nnunet_v1/checkpoints/phase3_best.pth \
        --output_dir viz_best_slice_seg

    # 端到端预测模式: whole brain -> loc(pred centroid) -> seg
    python visualize_best_slice.py \
        --config configs/default.yaml \
        --mode pred \
        --checkpoint runs/run_nnunet_v1/checkpoints/phase3_best.pth \
        --output_dir viz_best_slice_pred \
        --gpu 1

    # 最近一次 nnUNet 训练 (phase1_loc_opt_run2) 的 phase3 → OpenNeuro 外推验证
    python visualize_best_slice.py \
        --config configs/default.yaml \
        --split_mode openneuro \
        --prepared_dir ${TN_ROOT}/prepared_data \
        --checkpoint ${TN_ROOT}/nnUNet_data/nnUNet_results_phase1_loc_opt_run2/Dataset001_TN/nnUNetTrainerLoc__nnUNetPlans__3d_fullres/fold_0/phase3_best.pth \
        --mode pred \
        --output_dir viz_openneuro_generalization \
        --gpu 1
"""
import os
REPO_ROOT = os.environ.get('TN_ROOT', os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import yaml

from models.differentiable_crop import differentiable_crop_3d
from models.locseg_net import LocSegNet
from models.two_stage_loc import extract_roi_around_centroid, fine_norm_to_global_norm
from data.dataset import TNLocSegDataset, get_train_val_split
from nnunet_loc_trainer import nnUNetTrainerLoc
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw


def load_model(cfg, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "network_weights" in ckpt:
        return _load_nnunet_loc_model(cfg, ckpt, checkpoint_path, device)
    return _load_locseg_model(cfg, ckpt, checkpoint_path, device)


def _load_locseg_model(cfg, ckpt, checkpoint_path, device):
    seg_arch = cfg["segmentation"].get("seg_arch", "nnunet")
    loc_arch = cfg["localization"].get("loc_arch", "custom")
    model = LocSegNet(
        loc_input_size=tuple(cfg["localization"]["input_size"]),
        seg_crop_size=tuple(cfg["segmentation"]["crop_size"]),
        num_classes=cfg["segmentation"]["num_classes"],
        loc_channels=tuple(cfg["localization"]["encoder_channels"]),
        loc_arch=loc_arch,
        loc_features_per_stage=tuple(cfg["localization"].get("features_per_stage", [16, 32, 64, 128, 256])),
        loc_n_conv_per_stage=cfg["localization"].get("n_conv_per_stage", 2),
        loc_n_conv_per_stage_decoder=cfg["localization"].get("n_conv_per_stage_decoder", 2),
        seg_channels=tuple(cfg["segmentation"].get("encoder_channels", [32, 64, 128, 256, 512])),
        deep_supervision=cfg["segmentation"]["deep_supervision"],
        seg_arch=seg_arch,
        seg_features_per_stage=tuple(cfg["segmentation"].get("features_per_stage", [32, 64, 128, 256, 320])),
        n_conv_per_stage=cfg["segmentation"].get("n_conv_per_stage", 2),
        n_conv_per_stage_decoder=cfg["segmentation"].get("n_conv_per_stage_decoder", 2),
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    phase = ckpt.get("phase", "?")
    print(f"模型: {checkpoint_path} (phase={phase}, epoch={epoch}, loc_arch={loc_arch}, seg_arch={seg_arch})")
    return model


class _NnUNetLocVizAdapter(torch.nn.Module):
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
        else:
            c = 0.5 * (c_l + c_r)
        return {
            "heatmap_left": hm_l,
            "heatmap_right": hm_r,
            "centroid_left": c_l,
            "centroid_right": c_r,
            "centroid_norm": c,
        }

    def forward_seg(self, wb, centroid_norm):
        mod = self.net.module if hasattr(self.net, "module") else self.net
        crop_size = tuple(int(i) for i in self.trainer.configuration_manager.patch_size)
        roi = differentiable_crop_3d(wb, centroid_norm, crop_size, mode="bilinear")
        seg_out = mod.forward_seg(roi)
        return seg_out, None, roi


def _load_nnunet_loc_model(cfg, ckpt, checkpoint_path, device):
    init_args = ckpt["init_args"]
    trainer = nnUNetTrainerLoc(
        plans=init_args["plans"],
        configuration=init_args["configuration"],
        fold=init_args["fold"],
        dataset_json=init_args["dataset_json"],
        device=torch.device(device),
    )
    trainer.initialize()
    # 可视化只需要网络权重，避免因不同 phase 的优化器参数组不一致导致加载失败
    trainer.network.load_state_dict(ckpt["network_weights"])
    model = _NnUNetLocVizAdapter(trainer).to(device).eval()
    epoch = ckpt.get("current_epoch", "?")
    print(f"模型: {checkpoint_path} (nnUNetTrainerLoc, epoch={epoch})")
    return model


def center_crop_3d(vol, target_shape):
    """Center crop a 3D/5D volume."""
    if isinstance(vol, torch.Tensor) and vol.ndim == 5:
        _, _, d, h, w = vol.shape
        td, th, tw = target_shape
        d0, h0, w0 = (d - td) // 2, (h - th) // 2, (w - tw) // 2
        return vol[:, :, d0:d0+td, h0:h0+th, w0:w0+tw]
    d, h, w = vol.shape[:3]
    td, th, tw = target_shape
    d0, h0, w0 = (d - td) // 2, (h - th) // 2, (w - tw) // 2
    return vol[d0:d0+td, h0:h0+th, w0:w0+tw]


def crop_roi_from_whole_brain(whole_brain, centroid_vox, crop_shape):
    """
    以 centroid_vox 为中心, 从 whole_brain 裁剪 ROI.
    超出边界的区域补 0.
    """
    d, h, w = whole_brain.shape
    cd, ch, cw = np.round(centroid_vox).astype(int)
    td, th, tw = crop_shape
    hd, hh, hw = td // 2, th // 2, tw // 2

    start = np.array([cd - hd, ch - hh, cw - hw], dtype=int)
    end = start + np.array([td, th, tw], dtype=int)

    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, np.array([d, h, w], dtype=int))

    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    roi = np.zeros((td, th, tw), dtype=whole_brain.dtype)
    roi[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = whole_brain[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]
    return roi


def dice_2d(pred_slice, gt_slice, cls):
    p = (pred_slice == cls).astype(float)
    g = (gt_slice == cls).astype(float)
    inter = (p * g).sum()
    union = p.sum() + g.sum()
    if union == 0:
        return float('nan')
    return (2 * inter / union)


def find_best_slice(pred, gt, num_classes=3, slice_fg_filter="any"):
    """
    遍历 3 个轴的每个 slice, 计算前景类平均 Dice, 返回最佳 slice 信息.

    返回:
        best: dict with axis, slice_idx, mean_dice, per_class_dice
        all_dices: dict {axis: {cls: [dice_per_slice]}}
    """
    best = {"axis": 0, "slice_idx": 0, "mean_dice": -1, "per_class": {}}
    all_dices = {}

    for axis in range(3):
        n_slices = pred.shape[axis]
        axis_dices = {c: [] for c in range(1, num_classes)}

        for s in range(n_slices):
            if axis == 0:
                p_sl, g_sl = pred[s, :, :], gt[s, :, :]
            elif axis == 1:
                p_sl, g_sl = pred[:, s, :], gt[:, s, :]
            else:
                p_sl, g_sl = pred[:, :, s], gt[:, :, s]

            has_nerve = (g_sl == 1).any()
            has_vessel = (g_sl == 2).any()
            if slice_fg_filter == "both":
                if not (has_nerve and has_vessel):
                    for c in range(1, num_classes):
                        axis_dices[c].append(float("nan"))
                    continue
            else:
                if not (has_nerve or has_vessel):
                    for c in range(1, num_classes):
                        axis_dices[c].append(float("nan"))
                    continue

            slice_dices = {}
            for c in range(1, num_classes):
                d = dice_2d(p_sl, g_sl, c)
                axis_dices[c].append(d)
                slice_dices[c] = d

            valid = [v for v in slice_dices.values() if not np.isnan(v)]
            mean_d = np.mean(valid) if valid else -1

            if mean_d > best["mean_dice"]:
                best = {
                    "axis": axis,
                    "slice_idx": s,
                    "mean_dice": mean_d,
                    "per_class": slice_dices,
                }

        all_dices[axis] = axis_dices

    return best, all_dices


def compute_mean_dice_per_axis(all_dices, gt, num_classes=3, slice_fg_filter="any"):
    """
    对每个 axis:
      nerve 和 vessel 完全独立统计:
         - nerve 平均 Dice 只在 GT 含 nerve 的 slice 上计算
         - vessel 平均 Dice 只在 GT 含 vessel 的 slice 上计算

    返回:
        axis_mean: dict {
            axis: {
                "nerve": float, "vessel": float, "macro": float,
                "n_valid_nerve": int, "n_valid_vessel": int
            }
        }
        overall_mean: dict {"nerve": float, "vessel": float, "macro": float}
    """
    axis_mean = {}

    all_nerve, all_vessel = [], []

    for axis in range(3):
        n_slices = gt.shape[axis]
        nerve_dices, vessel_dices = [], []
        for s in range(n_slices):
            if axis == 0:
                g_sl = gt[s, :, :]
            elif axis == 1:
                g_sl = gt[:, s, :]
            else:
                g_sl = gt[:, :, s]

            has_nerve = (g_sl == 1).any()
            has_vessel = (g_sl == 2).any()
            if slice_fg_filter == "both":
                if not (has_nerve and has_vessel):
                    continue
            elif not has_nerve and not has_vessel:
                continue

            if has_nerve:
                nerve_dices.append(all_dices[axis][1][s])
            if has_vessel:
                vessel_dices.append(all_dices[axis][2][s])

        nerve_mean = np.nanmean(nerve_dices) if nerve_dices else float('nan')
        vessel_mean = np.nanmean(vessel_dices) if vessel_dices else float('nan')
        macro = np.nanmean([nerve_mean, vessel_mean])

        axis_mean[axis] = {
            "nerve": nerve_mean, "vessel": vessel_mean,
            "macro": macro,
            "n_valid_nerve": len(nerve_dices),
            "n_valid_vessel": len(vessel_dices),
        }
        all_nerve.extend(nerve_dices)
        all_vessel.extend(vessel_dices)

    overall_mean = {
        "nerve": np.nanmean(all_nerve) if all_nerve else float('nan'),
        "vessel": np.nanmean(all_vessel) if all_vessel else float('nan'),
        "macro": np.nanmean([
            np.nanmean(all_nerve) if all_nerve else float('nan'),
            np.nanmean(all_vessel) if all_vessel else float('nan'),
        ]),
    }

    return axis_mean, overall_mean


def predict_case(
    model,
    wb_tensor,
    centroid_norm,
    wb_shape,
    mask_shape,
    device,
    loc_two_stage: bool = False,
    fine_crop_size: int = 64,
):
    """
    端到端预测: whole brain → loc (预测 centroid) → seg.
    返回 pred mask, roi, 以及预测的 centroid 体素坐标.

    loc_two_stage: True 时先全脑粗定位，再在粗质心周围 fine_crop_size³ ROI 内二次定位（同一权重）。
    """
    with torch.no_grad():
        wb = wb_tensor.unsqueeze(0).to(device)
        if isinstance(centroid_norm, np.ndarray):
            gt_c = torch.from_numpy(centroid_norm).unsqueeze(0).float().to(device)
        else:
            gt_c = centroid_norm.unsqueeze(0).float().to(device)

        if loc_two_stage:
            fs = wb_shape.view(-1).float().to(device)
            loc_coarse = model.forward_loc(wb, gt_centroid_norm=gt_c)
            crop, starts, sizes = extract_roi_around_centroid(
                wb, loc_coarse["centroid_norm"], fine_crop_size, fs
            )
            loc_fine = model.forward_loc(crop, gt_centroid_norm=gt_c)
            pred_centroid_norm = fine_norm_to_global_norm(
                loc_fine["centroid_norm"], starts, sizes, fs
            )
        else:
            loc_result = model.forward_loc(wb, gt_centroid_norm=gt_c)
            pred_centroid_norm = loc_result["centroid_norm"]
        seg_out, _, roi = model.forward_seg(wb, pred_centroid_norm)
        if isinstance(seg_out, (list, tuple)):
            seg_out = seg_out[0]

        if isinstance(seg_out, dict):
            raise NotImplementedError("dual_seg not supported here")

        if seg_out.shape[2] > mask_shape[0]:
            seg_out = center_crop_3d(seg_out, mask_shape)

        pred = torch.argmax(seg_out, dim=1).squeeze(0).cpu().numpy()

        roi_np = roi.cpu().squeeze(0).squeeze(0).numpy()
        if roi_np.shape[0] > mask_shape[0]:
            roi_np = center_crop_3d(roi_np, mask_shape)

        pred_centroid_vox = (pred_centroid_norm.squeeze(0).cpu().numpy()
                             * (wb_shape - 1)).round().astype(int)

    return pred, roi_np, pred_centroid_vox


def predict_loc_only(
    model,
    wb_tensor,
    centroid_norm,
    wb_shape,
    device,
    two_stage: bool = False,
    fine_crop_size: int = 64,
):
    """
    仅定位: whole brain → loc → 预测质心体素坐标. 不调用 seg.
    与 predict_case 中 loc 部分一致, 用于只做 loc 的可视化/验证.

    two_stage: True 时为粗→细二次定位（同一模型权重，细定位在粗质心周围的立方 ROI 内）。
    """
    with torch.no_grad():
        wb = wb_tensor.unsqueeze(0).to(device)
        if isinstance(centroid_norm, np.ndarray):
            gt_c = torch.from_numpy(centroid_norm).unsqueeze(0).float().to(device)
        else:
            gt_c = centroid_norm.unsqueeze(0).float().to(device)

        if two_stage:
            fs = wb_shape.view(-1).float().to(device)
            loc_coarse = model.forward_loc(wb, gt_centroid_norm=gt_c)
            crop, starts, sizes = extract_roi_around_centroid(
                wb, loc_coarse["centroid_norm"], fine_crop_size, fs
            )
            loc_fine = model.forward_loc(crop, gt_centroid_norm=gt_c)
            pred_centroid_norm = fine_norm_to_global_norm(
                loc_fine["centroid_norm"], starts, sizes, fs
            )
        else:
            loc_result = model.forward_loc(wb, gt_centroid_norm=gt_c)
            pred_centroid_norm = loc_result["centroid_norm"]
        wbs = wb_shape.cpu().numpy() if isinstance(wb_shape, torch.Tensor) else np.asarray(wb_shape)
        pred_centroid_vox = (pred_centroid_norm.squeeze(0).cpu().numpy()
                             * (wbs - 1)).round().astype(int)

    return pred_centroid_vox


def predict_case_seg_only(model, wb_tensor, centroid_norm, mask_shape, device):
    """
    仅分割预测: 使用 GT centroid 直接裁剪 ROI 后做 seg.
    不经过 loc 分支.
    """
    with torch.no_grad():
        wb = wb_tensor.unsqueeze(0).to(device)
        if isinstance(centroid_norm, np.ndarray):
            gt_c = torch.from_numpy(centroid_norm).unsqueeze(0).float().to(device)
        else:
            gt_c = centroid_norm.unsqueeze(0).float().to(device)

        seg_out, _, roi = model.forward_seg(wb, gt_c)
        if isinstance(seg_out, (list, tuple)):
            seg_out = seg_out[0]

        if isinstance(seg_out, dict):
            raise NotImplementedError("dual_seg not supported here")

        if seg_out.shape[2] > mask_shape[0]:
            seg_out = center_crop_3d(seg_out, mask_shape)

        pred = torch.argmax(seg_out, dim=1).squeeze(0).cpu().numpy()

        roi_np = roi.cpu().squeeze(0).squeeze(0).numpy()
        if roi_np.shape[0] > mask_shape[0]:
            roi_np = center_crop_3d(roi_np, mask_shape)

    return pred, roi_np


def align_masks_in_brain_space(gt_mask, pred_mask, gt_centroid_vox, pred_centroid_vox,
                               roi=None):
    """
    将 GT 和 pred mask 放入全脑坐标系中对齐, 返回重叠区域的两个 mask.

    GT mask 中心在 gt_centroid_vox, pred mask 中心在 pred_centroid_vox,
    两者都是 (ms, ms, ms) 大小.

    返回:
        aligned_gt, aligned_pred: 对齐后的 mask (在重叠区域), shape 相同
        aligned_roi: 对齐后的 ROI (如果提供了 roi)
        None, None, None 如果无重叠
    """
    ms = gt_mask.shape[0]
    half = ms // 2

    gt_start = gt_centroid_vox - half
    gt_end = gt_start + ms
    pred_start = pred_centroid_vox - half
    pred_end = pred_start + ms

    overlap_start = np.maximum(gt_start, pred_start)
    overlap_end = np.minimum(gt_end, pred_end)
    overlap_shape = overlap_end - overlap_start

    if (overlap_shape <= 0).any():
        return None, None, None

    gt_local_start = overlap_start - gt_start
    gt_local_end = gt_local_start + overlap_shape
    pred_local_start = overlap_start - pred_start
    pred_local_end = pred_local_start + overlap_shape

    aligned_gt = gt_mask[
        gt_local_start[0]:gt_local_end[0],
        gt_local_start[1]:gt_local_end[1],
        gt_local_start[2]:gt_local_end[2],
    ]
    aligned_pred = pred_mask[
        pred_local_start[0]:pred_local_end[0],
        pred_local_start[1]:pred_local_end[1],
        pred_local_start[2]:pred_local_end[2],
    ]

    aligned_roi = None
    if roi is not None:
        aligned_roi = roi[
            pred_local_start[0]:pred_local_end[0],
            pred_local_start[1]:pred_local_end[1],
            pred_local_start[2]:pred_local_end[2],
        ]

    return aligned_gt, aligned_pred, aligned_roi


def make_label_rgb(label, roi_slice=None):
    """label → RGB, 可选叠加在 ROI 灰度上."""
    if roi_slice is not None:
        vmin, vmax = np.percentile(roi_slice[roi_slice > 0], [1, 99]) if (roi_slice > 0).any() else (0, 1)
        gray = np.clip((roi_slice - vmin) / (vmax - vmin + 1e-8), 0, 1)
        rgb = np.stack([gray, gray, gray], axis=-1)
    else:
        rgb = np.zeros((*label.shape, 3))

    nerve = label == 1
    vessel = label == 2
    if nerve.any():
        rgb[nerve] = rgb[nerve] * 0.3 + np.array([1.0, 0.2, 0.2]) * 0.7
    if vessel.any():
        rgb[vessel] = rgb[vessel] * 0.3 + np.array([0.3, 0.5, 1.0]) * 0.7
    return rgb


def visualize_case(case_id, pred, gt, roi, best, all_dices, output_dir, num_classes=3):
    axis = best["axis"]
    s = best["slice_idx"]
    axis_names = ["Dim0 (Sagittal)", "Dim1 (Coronal)", "Dim2 (Axial)"]

    if axis == 0:
        roi_sl = roi[s, :, :]
        gt_sl = gt[s, :, :]
        pred_sl = pred[s, :, :]
    elif axis == 1:
        roi_sl = roi[:, s, :]
        gt_sl = gt[:, s, :]
        pred_sl = pred[:, s, :]
    else:
        roi_sl = roi[:, :, s]
        gt_sl = gt[:, :, s]
        pred_sl = pred[:, :, s]

    # 对比图: TP=绿, FP=黄, FN=紫
    diff = np.zeros((*gt_sl.shape, 3))
    vmin, vmax = np.percentile(roi_sl[roi_sl > 0], [1, 99]) if (roi_sl > 0).any() else (0, 1)
    gray = np.clip((roi_sl - vmin) / (vmax - vmin + 1e-8), 0, 1)
    diff = np.stack([gray, gray, gray], axis=-1) * 0.4

    for c in range(1, num_classes):
        tp = (pred_sl == c) & (gt_sl == c)
        fp = (pred_sl == c) & (gt_sl != c)
        fn = (pred_sl != c) & (gt_sl == c)
        diff[tp] = [0, 1, 0]       # green = correct
        diff[fp] = [1, 1, 0]       # yellow = false positive
        diff[fn] = [0.8, 0, 0.8]   # purple = false negative

    # --- Plot ---
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)

    class_names = {1: "Nerve", 2: "Vessel"}
    dice_str = ", ".join(
        f"{class_names.get(c, f'C{c}')}: {best['per_class'].get(c, 0):.4f}"
        for c in range(1, num_classes)
    )
    fig.suptitle(
        f"{case_id}  |  Best slice: {axis_names[axis]} #{s}  |  "
        f"Mean Dice={best['mean_dice']:.4f}  |  {dice_str}",
        fontsize=13, y=0.98,
    )

    # 上排: ROI | GT | Pred | Diff
    titles = ["ROI", "GT", "Prediction", "Comparison (G=TP, Y=FP, P=FN)"]
    images = [
        np.stack([np.clip((roi_sl.T - vmin)/(vmax-vmin+1e-8), 0, 1)]*3, axis=-1),
        make_label_rgb(gt_sl.T, roi_sl.T),
        make_label_rgb(pred_sl.T, roi_sl.T),
        diff.transpose(1, 0, 2),
    ]

    for col in range(4):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(images[col], origin="lower")
        ax.set_title(titles[col], fontsize=11)
        ax.axis("off")

    # 下排: 3 个轴的 Dice 曲线
    colors = {1: "#FF4444", 2: "#4488FF"}
    for plot_axis in range(3):
        ax = fig.add_subplot(gs[1, plot_axis])
        for c in range(1, num_classes):
            dices = all_dices[plot_axis][c]
            valid_x = [i for i, d in enumerate(dices) if not np.isnan(d)]
            valid_d = [d for d in dices if not np.isnan(d)]
            ax.plot(valid_x, valid_d, color=colors[c], linewidth=1.2,
                    label=f"{class_names.get(c, f'C{c}')}", alpha=0.8)
        if plot_axis == axis:
            ax.axvline(s, color="lime", linewidth=2, linestyle="--",
                       label=f"Best slice ({s})", alpha=0.8)
        ax.set_title(f"{axis_names[plot_axis]}", fontsize=10)
        ax.set_xlabel("Slice index")
        ax.set_ylabel("2D Dice")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 右下角: 汇总信息
    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    summary = (
        f"Case: {case_id}\n\n"
        f"Best axis: {axis_names[axis]}\n"
        f"Best slice: {s}\n"
        f"Mean Dice: {best['mean_dice']:.4f}\n\n"
    )
    for c in range(1, num_classes):
        d = best["per_class"].get(c, float('nan'))
        summary += f"{class_names.get(c, f'C{c}')}: {d:.4f}\n"

    # 全局 3D Dice
    summary += "\n--- 3D Volume Dice ---\n"
    for c in range(1, num_classes):
        p = (pred == c).astype(float)
        g = (gt == c).astype(float)
        inter = (p * g).sum()
        union = p.sum() + g.sum()
        d3d = 2 * inter / union if union > 0 else 0
        summary += f"{class_names.get(c, f'C{c}')}: {d3d:.4f}\n"

    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    out_path = os.path.join(output_dir, f"{case_id}_best.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    return best["mean_dice"]


def _load_val_ids_from_nnunet_split(dataset_name: str, fold: int):
    if nnUNet_preprocessed is None or nnUNet_raw is None:
        raise RuntimeError("nnUNet_preprocessed/nnUNet_raw 未设置，无法读取 nnU-Net 标准划分。")

    splits_file = os.path.join(nnUNet_preprocessed, dataset_name, "splits_final.json")
    mapping_file = os.path.join(nnUNet_raw, dataset_name, "case_mapping.json")
    if not os.path.isfile(splits_file):
        raise FileNotFoundError(f"未找到 splits_final.json: {splits_file}")
    if not os.path.isfile(mapping_file):
        raise FileNotFoundError(f"未找到 case_mapping.json: {mapping_file}")

    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    if fold < 0 or fold >= len(splits):
        raise ValueError(f"fold={fold} 越界，可用范围 [0, {len(splits)-1}]")
    val_keys = splits[fold]["val"]

    with open(mapping_file, "r", encoding="utf-8") as f:
        key2prepared = json.load(f)

    val_ids, missing = [], []
    for k in val_keys:
        p = key2prepared.get(k, None)
        if p is None:
            missing.append(k)
        else:
            val_ids.append(p)

    if missing:
        print(f"[WARN] {len(missing)} 个 key 在 case_mapping 中缺失，将跳过。")
    print(f"[split] 使用 nnU-Net fold={fold}: val keys={len(val_keys)} -> prepared cases={len(val_ids)}")
    return val_ids


def _load_openneuro_case_ids(prepared_dir: str) -> list:
    """
    从 merged prepared_data 中筛出 OpenNeuro 来源的 case（info.json 含 OpenNeuro 或目录名 sub-*）。
    用于：院内训练 checkpoint 在未参与训练的 OpenNeuro 数据上测泛化/鲁棒性。
    """
    meta_path = os.path.join(prepared_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"未找到 metadata.json: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    case_ids = meta.get("case_ids", [])
    out = []
    for cid in case_ids:
        info_path = os.path.join(prepared_dir, cid, "info.json")
        if not os.path.isfile(info_path):
            continue
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        src = info.get("source", "") or ""
        if "OpenNeuro" in src or cid.startswith("sub-"):
            out.append(cid)
    out.sort()
    print(f"[split] openneuro 外推: prepared_dir={prepared_dir}, cases={len(out)}")
    if not out:
        print("[WARN] 未找到任何 OpenNeuro case；请确认 prepared_dir 含 merged 数据或改用 prepared_data_openneuro。")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="runs/run_nnunet_v1/checkpoints/phase3_best.pth")
    parser.add_argument("--output_dir", default="viz_best_slice")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--mode", choices=["seg", "pred", "gt"], default="seg",
                        help="seg: GT centroid裁剪+seg; pred: loc->seg; gt: pred=gt")
    parser.add_argument("--num_cases", type=int, default=0,
                        help="0=全部验证集")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", type=str, default="Dataset001_TN",
                        help="读取 nnU-Net 划分时使用的数据集名")
    parser.add_argument("--fold", type=int, default=0,
                        help="读取 nnU-Net 标准划分时的 fold")
    parser.add_argument("--prepared_dir", type=str, default=None,
                        help="覆盖 config 中 data.prepared_dir（例如 merged prepared_data 上的 openneuro 子集）")
    parser.add_argument("--split_mode", choices=["nnunet", "random", "openneuro"], default="nnunet",
                        help="nnunet: splits_final; random: 随机划分 val; openneuro: 仅 OpenNeuro case（外推鲁棒性）")
    parser.add_argument("--slice_fg_filter", choices=["any", "both"], default="any",
                        help="any: slice 有任一前景就统计; both: 仅统计同时含 nerve+vessel 的 slice")
    parser.add_argument("--loc_two_stage", action="store_true",
                        help="pred 模式: 全脑粗定位后在 ROI 内二次定位再分割（同一 checkpoint）")
    parser.add_argument("--fine_crop_size", type=int, default=64,
                        help="二次定位时围绕粗质心的立方 ROI 边长（体素），默认 64")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    model = None
    if args.mode in ("pred", "seg"):
        model = load_model(cfg, args.checkpoint, device)
    else:
        print("模式: GT 分割 (pred=gt), 不加载模型。")

    prepared_dir = args.prepared_dir or cfg["data"]["prepared_dir"]
    if args.split_mode == "nnunet":
        val_ids = _load_val_ids_from_nnunet_split(args.dataset_name, args.fold)
    elif args.split_mode == "openneuro":
        val_ids = _load_openneuro_case_ids(prepared_dir)
    else:
        _, val_ids = get_train_val_split(
            prepared_dir,
            train_ratio=cfg["data"]["train_val_split"],
            seed=cfg["data"]["random_seed"],
        )
        print(f"[split] 使用随机切分: val={len(val_ids)}")
    mask_size = cfg["segmentation"].get("mask_size", None)
    val_dataset = TNLocSegDataset(
        prepared_dir, case_ids=val_ids, phase=2,
        cache_wb=False, lr_flip_prob=0.0, mask_size=mask_size,
    )

    if args.num_cases > 0:
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(val_dataset), size=min(args.num_cases, len(val_dataset)), replace=False)
    else:
        indices = range(len(val_dataset))

    loc_mode = (
        f"two_stage (fine_crop={args.fine_crop_size})"
        if args.loc_two_stage
        else "single_stage"
    )
    print(f"评估 {len(indices)} 个 case → {args.output_dir}/  [loc: {loc_mode}]\n")

    mask_shape_3d = (mask_size, mask_size, mask_size) if mask_size else tuple(cfg["segmentation"]["crop_size"])
    num_classes = cfg["segmentation"]["num_classes"]
    all_case_overall = []  # per-case overall mean dice (filtered)
    all_loc_err = []  # (case_id, err_voxels) for mode=pred
    axis_names = ["Dim0 (Sagittal)", "Dim1 (Coronal)", "Dim2 (Axial)"]

    for i in indices:
        sample = val_dataset[i]
        case_id = sample["case_id"]
        print(f"[{case_id}]")

        wb_tensor = sample["whole_brain"]
        centroid_norm = sample["centroid_norm"]
        gt_mask = sample["mask"].numpy()
        gt_centroid_vox = sample["centroid_ras"].numpy().round().astype(int)
        wb_shape = sample["wb_shape"].numpy()

        if args.mode == "pred":
            pred, roi, pred_centroid_vox = predict_case(
                model,
                wb_tensor,
                centroid_norm,
                wb_shape,
                mask_shape_3d,
                device,
                loc_two_stage=args.loc_two_stage,
                fine_crop_size=args.fine_crop_size,
            )
            loc_err = float(np.linalg.norm(gt_centroid_vox - pred_centroid_vox))
            all_loc_err.append((case_id, loc_err))
            print(f"  Loc error: {loc_err:.2f} voxels  "
                  f"(GT={gt_centroid_vox}, Pred={pred_centroid_vox})")

            aligned_gt, aligned_pred, aligned_roi = align_masks_in_brain_space(
                gt_mask, pred, gt_centroid_vox, pred_centroid_vox, roi=roi)
        elif args.mode == "seg":
            pred, roi = predict_case_seg_only(
                model, wb_tensor, centroid_norm, mask_shape_3d, device)
            aligned_gt, aligned_pred, aligned_roi = gt_mask, pred, roi
        else:
            pred = gt_mask.copy()
            pred_centroid_vox = gt_centroid_vox.copy()
            wb_np = wb_tensor.squeeze(0).numpy() if wb_tensor.ndim == 4 else wb_tensor.numpy()
            roi = crop_roi_from_whole_brain(wb_np, gt_centroid_vox, mask_shape_3d)
            aligned_gt, aligned_pred, aligned_roi = gt_mask, pred, roi

        if aligned_gt is None:
            print(f"  WARNING: no overlap between GT and pred crops! Skipping.")
            continue

        overlap_pct = np.prod(aligned_gt.shape) / np.prod(gt_mask.shape) * 100
        print(f"  Overlap: {aligned_gt.shape} ({overlap_pct:.0f}% of {mask_shape_3d})")

        best, all_dices = find_best_slice(
            aligned_pred, aligned_gt, num_classes=num_classes, slice_fg_filter=args.slice_fg_filter
        )
        if best["mean_dice"] < 0:
            print("  WARNING: 当前筛选条件下没有有效 slice，跳过该 case。")
            continue
        axis_mean, overall_mean = compute_mean_dice_per_axis(
            all_dices, aligned_gt, num_classes=num_classes, slice_fg_filter=args.slice_fg_filter)

        print(f"  Vis slice(best for display): axis={best['axis']}, idx={best['slice_idx']}, "
              f"dice={best['mean_dice']:.4f}")
        for ax in range(3):
            am = axis_mean[ax]
            print(f"  {axis_names[ax]:20s}: Nerve={am['nerve']:.4f}  Vessel={am['vessel']:.4f}  "
                  f"Macro={am['macro']:.4f}  "
                  f"(n_nerve={am['n_valid_nerve']}, n_vessel={am['n_valid_vessel']})")
        print(f"  Overall (filtered):   Nerve={overall_mean['nerve']:.4f}  "
              f"Vessel={overall_mean['vessel']:.4f}  Macro={overall_mean['macro']:.4f}")

        visualize_case(case_id, aligned_pred, aligned_gt, aligned_roi, best, all_dices,
                       args.output_dir, num_classes=num_classes)
        all_case_overall.append(overall_mean)

    if all_case_overall:
        print(f"\n{'='*60}")
        if args.slice_fg_filter == "both":
            print(f"Filtered avg 2D Dice ({len(all_case_overall)} cases, slices with BOTH foregrounds):")
        else:
            print(f"Filtered avg 2D Dice ({len(all_case_overall)} cases, slices with any foreground):")
        nerve_all = [c["nerve"] for c in all_case_overall if not np.isnan(c["nerve"])]
        vessel_all = [c["vessel"] for c in all_case_overall if not np.isnan(c["vessel"])]
        macro_all = [c["macro"] for c in all_case_overall if not np.isnan(c["macro"])]
        print(f"  Nerve  avg: {np.mean(nerve_all):.4f}  (median {np.median(nerve_all):.4f})")
        print(f"  Vessel avg: {np.mean(vessel_all):.4f}  (median {np.median(vessel_all):.4f})")
        print(f"  Macro:      {np.mean(macro_all):.4f}  (median {np.median(macro_all):.4f})")

    if all_loc_err:
        errs = np.array([e for _, e in all_loc_err], dtype=np.float64)
        print(f"\n{'='*60}")
        print(f"Loc error 汇总 (体素 L2, 全脑坐标, n={len(all_loc_err)}):")
        print(f"  mean={errs.mean():.2f}  median={np.median(errs):.2f}  "
              f"min={errs.min():.2f}  max={errs.max():.2f}  std={errs.std():.2f}")
        print("  per-case:")
        for cid, e in sorted(all_loc_err, key=lambda x: x[0]):
            print(f"    {cid}: {e:.2f}")


if __name__ == "__main__":
    main()
