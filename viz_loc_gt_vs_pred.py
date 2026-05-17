#!/usr/bin/env python3
"""
在三个正交切片上叠加 GT / Pred 质心，并标注各维体素差 (pred - gt)。

仅运行定位分支 (forward_loc)，不调用分割，适合 Phase1 权重或只做 loc 验证。

用法示例:
  CUDA_VISIBLE_DEVICES=1 conda run -n tnproject python viz_loc_gt_vs_pred.py \\
    --config configs/default.yaml \\
    --checkpoint nnUNet_data/nnUNet_results_phase1_loc_opt_run2/.../phase1_best.pth \\
    --prepared_dir prepared_data --split_mode openneuro \\
    --num_cases 5 --output_dir viz_loc_compare --gpu 0

  # 二次定位（粗→细，同一权重）:
  ... --two_stage --fine_crop_size 64
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data.dataset import TNLocSegDataset, get_train_val_split
from visualize_best_slice import (
    _load_openneuro_case_ids,
    _load_val_ids_from_nnunet_split,
    load_model,
    predict_loc_only,
)


def _clip_vox(v, shape):
    v = np.asarray(v, dtype=np.float64).round().astype(np.int64)
    return np.clip(v, 0, np.maximum(shape - 1, 0))


def plot_loc_three_planes(
    vol_3d: np.ndarray,
    gt_vox: np.ndarray,
    pred_vox: np.ndarray,
    case_id: str,
    out_path: str,
):
    """
    vol_3d: (D0, D1, D2) 与 dataset whole_brain 一致.
    切片取在 GT 质心处；Pred 投影到同一平面显示偏移。
    """
    d0, d1, d2 = vol_3d.shape
    g = _clip_vox(gt_vox, np.array([d0, d1, d2]))
    p = _clip_vox(pred_vox, np.array([d0, d1, d2]))
    delta = pred_vox.astype(np.float64) - gt_vox.astype(np.float64)
    l2 = float(np.linalg.norm(pred_vox.astype(np.float64) - gt_vox.astype(np.float64)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    titles = [
        (f"dim0={g[0]} fixed (plane dim1 vs dim2)", g[0], lambda: vol_3d[g[0], :, :], (1, 2)),
        (f"dim1={g[1]} fixed (plane dim0 vs dim2)", g[1], lambda: vol_3d[:, g[1], :], (0, 2)),
        (f"dim2={g[2]} fixed (plane dim0 vs dim1)", g[2], lambda: vol_3d[:, :, g[2]], (0, 1)),
    ]

    for ax, (title, _fix, get_sl, (ia, ib)) in zip(axes, titles):
        sl = get_sl()
        # imshow: 横轴 = 第二维索引, 纵轴 = 第一维索引 (origin lower)
        ax.imshow(sl, cmap="gray", origin="lower", aspect="auto")
        # 在当前平面内: 横轴对应 ib, 纵轴对应 ia
        ax.scatter(
            g[ib], g[ia], c="lime", s=140, marker="+", linewidths=2.2,
            label="GT", zorder=5,
        )
        ax.scatter(
            p[ib], p[ia], c="red", s=140, marker="+", linewidths=2.2,
            label="Pred", zorder=5,
        )
        ax.set_xlabel(f"index dim{ib}")
        ax.set_ylabel(f"index dim{ia}")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)

    supt = (
        f"{case_id}  |  d0={delta[0]:+.1f} d1={delta[1]:+.1f} d2={delta[2]:+.1f}  "
        f"(pred-gt, vox)  |  L2={l2:.2f}"
    )
    fig.suptitle(supt, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output_dir", default="viz_loc_compare")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_cases", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset_name", default="Dataset001_TN")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--split_mode", choices=["nnunet", "random", "openneuro"], default="openneuro")
    p.add_argument("--prepared_dir", default=None)
    p.add_argument("--two_stage", action="store_true",
                   help="全脑粗定位后在粗质心周围立方 ROI 内再跑一次定位（同一模型）")
    p.add_argument("--fine_crop_size", type=int, default=64,
                   help="细定位 ROI 立方边长（体素）")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(cfg, args.checkpoint, device)
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

    ds = TNLocSegDataset(
        prepared_dir,
        case_ids=val_ids,
        phase=1,
        cache_wb=False,
        lr_flip_prob=0.0,
        mask_size=None,
    )

    rng = np.random.RandomState(args.seed)
    n = min(args.num_cases, len(ds))
    if n <= 0:
        print("无可用 case")
        return
    idxs = rng.choice(len(ds), size=n, replace=False) if len(ds) > n else np.arange(n)

    mode_s = f"two_stage ROI={args.fine_crop_size}" if args.two_stage else "single_stage"
    print(f"输出目录: {args.output_dir}  (n={n}, loc={mode_s})\n")

    for ii in idxs:
        sample = ds[int(ii)]
        case_id = sample["case_id"]
        wb_tensor = sample["whole_brain"]
        centroid_norm = sample["centroid_norm"]
        gt_vox = sample["centroid_ras"].numpy()
        wb_shape = sample["wb_shape"].numpy()

        pred_vox = predict_loc_only(
            model,
            wb_tensor,
            centroid_norm,
            wb_shape,
            device,
            two_stage=args.two_stage,
            fine_crop_size=args.fine_crop_size,
        )

        wb_np = wb_tensor.squeeze(0).numpy() if wb_tensor.ndim == 4 else wb_tensor.numpy()
        delta = pred_vox.astype(np.float64) - gt_vox.astype(np.float64)
        l2 = float(np.linalg.norm(pred_vox.astype(np.float64) - gt_vox.astype(np.float64)))
        print(
            f"[{case_id}] GT_vox=({gt_vox[0]:.0f},{gt_vox[1]:.0f},{gt_vox[2]:.0f})  "
            f"Pred_vox=({pred_vox[0]:.0f},{pred_vox[1]:.0f},{pred_vox[2]:.0f})  "
            f"Δ=({delta[0]:+.1f},{delta[1]:+.1f},{delta[2]:+.1f})  L2={l2:.2f}"
        )

        out_png = os.path.join(args.output_dir, f"{case_id}_loc_gt_pred.png")
        plot_loc_three_planes(wb_np, gt_vox, pred_vox, case_id, out_png)
        print(f"  -> {out_png}\n")


if __name__ == "__main__":
    main()
