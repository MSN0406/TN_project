#!/usr/bin/env python3
"""
从 verification_results.csv + prepared_data 生成 2×3 综合图:
  上行: 仪表板前三个子图 — 误差直方图、CDF(mm)、Ipsi vs Contra 箱线图
  下行: 三个病例在同一解剖维度上的单切片 (dim0/1/2 由 --example_dim 指定, 默认 2=轴位) + GT/Pred

用法:
  python export_verification_dashboard_subset.py \\
    --csv verify_results_dual_head_100ep/verification_results.csv \\
    --example_dim 2
  # --example_dim: 0=矢状 1=冠状 2=轴位(默认); 第二行三例均用该维度
"""
import os
REPO_ROOT = os.environ.get('TN_ROOT', os.path.dirname(os.path.abspath(__file__)))
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.gridspec import GridSpec


def _normalize_image(img):
    if (img > 0).any():
        vmin, vmax = np.percentile(img[img > 0], [1, 99])
    else:
        vmin, vmax = img.min(), img.max()
    return np.clip((img - vmin) / (vmax - vmin + 1e-8), 0, 1)


def load_results(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "case_id": row["case_id"],
                    "side": row["side"],
                    "error_vox": float(row["error_vox"]),
                    "error_mm": float(row["error_mm"]),
                    "gt_d": float(row["gt_d"]),
                    "gt_h": float(row["gt_h"]),
                    "gt_w": float(row["gt_w"]),
                    "pred_d": float(row["pred_d"]),
                    "pred_h": float(row["pred_h"]),
                    "pred_w": float(row["pred_w"]),
                }
            )
    return rows


def load_wb_for_case(prepared_dir, case_id):
    info_path = os.path.join(prepared_dir, case_id, "info.json")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(info_path)
    with open(info_path) as f:
        info = json.load(f)
    nii_path = info["nii_path"]
    nii = nib.load(nii_path)
    data = nii.get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data[:, :, :, 0]
    return data


def plot_single_dim_slice(
    ax,
    wb_norm,
    gt_vox,
    pred_vox,
    dim_idx,
    case_id,
    error_mm,
    show_legend=False,
):
    """与 verify_localization.visualize_single_case 行1 一致: 单维度 2D 切片 + 标记."""
    gt = gt_vox.astype(int)
    pred = pred_vox
    view_names = ["Sagittal (dim0)", "Coronal (dim1)", "Axial (dim2)"]
    slices_wb = [
        wb_norm[np.clip(gt[0], 0, wb_norm.shape[0] - 1), :, :],
        wb_norm[:, np.clip(gt[1], 0, wb_norm.shape[1] - 1), :],
        wb_norm[:, :, np.clip(gt[2], 0, wb_norm.shape[2] - 1)],
    ]
    markers = [
        ((gt[2], gt[1]), (pred[2], pred[1])),
        ((gt[2], gt[0]), (pred[2], pred[0])),
        ((gt[1], gt[0]), (pred[1], pred[0])),
    ]
    sl = slices_wb[dim_idx]
    gt_xy, pred_xy = markers[dim_idx]
    ax.imshow(sl, cmap="gray", origin="lower", aspect="auto")
    ax.plot(*gt_xy, "r+", markersize=14, markeredgewidth=2.2, label="GT")
    ax.plot(*pred_xy, "c*", markersize=11, markeredgewidth=1.5, label="Pred")
    ax.plot(
        [gt_xy[0], pred_xy[0]],
        [gt_xy[1], pred_xy[1]],
        "y--",
        linewidth=1.0,
        alpha=0.75,
    )
    ax.set_title(
        f"{view_names[dim_idx]}\n{case_id}  |  err={error_mm:.2f} mm",
        fontsize=10,
    )
    ax.axis("off")
    if show_legend:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)


def pick_three_cases(rows_sorted):
    """按误差分位取低/中/高各一例. rows_sorted: 按 error_mm 升序."""
    n = len(rows_sorted)
    if n == 0:
        return []
    cand = [n // 4, n // 2, 3 * n // 4]
    cand = [min(n - 1, max(0, i)) for i in cand]
    idxs = []
    for i in cand:
        if i not in idxs:
            idxs.append(i)
    k = 0
    while len(idxs) < 3 and k < n:
        if k not in idxs:
            idxs.append(k)
        k += 1
    return [rows_sorted[i] for i in idxs[:3]]


def plot_dashboard_2x3(
    results,
    prepared_dir,
    out_path,
    ckpt_name="",
    example_dim=2,
):
    errors_vox = np.array([r["error_vox"] for r in results])
    errors_mm = np.array([r["error_mm"] for r in results])

    dim_idx = int(example_dim)
    if dim_idx not in (0, 1, 2):
        raise ValueError("example_dim 必须为 0、1 或 2")
    view_names = ["Sagittal (dim0)", "Coronal (dim1)", "Axial (dim2)"]

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.05, 1.0], hspace=0.38, wspace=0.22)
    fig.suptitle(
        f"Localization Verification  (2×3)  |  {len(results)} cases  |  {ckpt_name}\n"
        f"Mean: {errors_vox.mean():.1f} vox / {errors_mm.mean():.2f} mm   "
        f"Median: {np.median(errors_vox):.1f} vox / {np.median(errors_mm):.2f} mm\n"
        f"Row 2 — three cases, same plane: {view_names[dim_idx]}",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    # ---- Row 0: (0,0) histogram ----
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(errors_vox, bins=25, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(errors_vox.mean(), color="red", ls="--", lw=2,
               label=f"Mean: {errors_vox.mean():.1f} vox")
    ax.axvline(np.median(errors_vox), color="orange", ls="--", lw=2,
               label=f"Median: {np.median(errors_vox):.1f} vox")
    ax.set_xlabel("Localization Error (voxels)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Error Distribution (Voxel)", fontsize=12)
    ax.legend(fontsize=9)
    ax2 = ax.twiny()
    ax2.hist(errors_mm, bins=25, alpha=0)
    ax2.set_xlabel("(mm)", fontsize=9, color="gray")
    ax2.tick_params(axis="x", labelsize=8, colors="gray")

    # ---- (0,1) CDF ----
    ax = fig.add_subplot(gs[0, 1])
    sorted_mm = np.sort(errors_mm)
    cdf = np.arange(1, len(sorted_mm) + 1) / len(sorted_mm) * 100
    ax.plot(sorted_mm, cdf, "b-", linewidth=2)
    ax.fill_between(sorted_mm, 0, cdf, alpha=0.15, color="blue")
    for thr_mm, color in [(2.0, "green"), (5.0, "orange"), (10.0, "red")]:
        pct = (errors_mm < thr_mm).mean() * 100
        ax.axvline(thr_mm, color=color, ls=":", lw=1.5, alpha=0.7)
        ax.annotate(
            f"<{thr_mm}mm: {pct:.0f}%",
            xy=(thr_mm, pct),
            fontsize=9,
            xytext=(thr_mm + 0.3, max(pct - 8, 5)),
            color=color,
            fontweight="bold",
        )
    ax.set_xlabel("Error (mm)", fontsize=11)
    ax.set_ylabel("Cumulative %", fontsize=11)
    ax.set_title("CDF (Physical Distance)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # ---- (0,2) Ipsi vs Contra ----
    ax = fig.add_subplot(gs[0, 2])
    ipsi_mm = [r["error_mm"] for r in results if r["side"] == "ipsi"]
    contra_mm = [r["error_mm"] for r in results if r["side"] == "contra"]
    data_box, labels_box = [], []
    if ipsi_mm:
        data_box.append(ipsi_mm)
        labels_box.append(
            f"Ipsi (n={len(ipsi_mm)})\nMean={np.mean(ipsi_mm):.2f}mm"
        )
    if contra_mm:
        data_box.append(contra_mm)
        labels_box.append(
            f"Contra (n={len(contra_mm)})\nMean={np.mean(contra_mm):.2f}mm"
        )
    if data_box:
        bp = ax.boxplot(
            data_box, tick_labels=labels_box, patch_artist=True, widths=0.5
        )
        colors_box = ["#3498db", "#e74c3c"]
        for patch, c in zip(bp["boxes"], colors_box[: len(data_box)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
    ax.set_ylabel("Error (mm)", fontsize=11)
    ax.set_title("Ipsi vs Contra (mm)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # ---- Row 1: 三例、同一维度 ----
    rows_by_err = sorted(results, key=lambda x: x["error_mm"])
    triple = pick_three_cases(rows_by_err)
    for col, row in enumerate(triple):
        ax = fig.add_subplot(gs[1, col])
        gt_vox = np.array([row["gt_d"], row["gt_h"], row["gt_w"]], dtype=np.float32)
        pred_vox = np.array([row["pred_d"], row["pred_h"], row["pred_w"]], dtype=np.float32)
        try:
            wb_raw = load_wb_for_case(prepared_dir, row["case_id"])
            wb_norm = _normalize_image(wb_raw)
            plot_single_dim_slice(
                ax,
                wb_norm,
                gt_vox,
                pred_vox,
                dim_idx,
                row["case_id"],
                row["error_mm"],
                show_legend=(col == 0),
            )
        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"无法加载体数据:\n{row['case_id']}\n{e!s}",
                ha="center",
                va="center",
                fontsize=9,
                transform=ax.transAxes,
                wrap=True,
            )
            ax.axis("off")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_top2_and_3d_legacy(results, out_dir, ckpt_name=""):
    """保留: 仅统计图 + 3D 的旧版输出 (可选)."""
    errors_vox = np.array([r["error_vox"] for r in results])
    errors_mm = np.array([r["error_mm"] for r in results])
    gt_all = np.array([[r["gt_d"], r["gt_h"], r["gt_w"]] for r in results])
    pred_all = np.array([[r["pred_d"], r["pred_h"], r["pred_w"]] for r in results])
    offsets = pred_all - gt_all

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.28)
    fig.suptitle(
        f"Localization Summary  |  {len(results)} cases  |  {ckpt_name}\n"
        f"Mean: {errors_vox.mean():.1f} vox / {errors_mm.mean():.2f} mm   "
        f"Median: {np.median(errors_vox):.1f} vox / {np.median(errors_mm):.2f} mm",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(errors_vox, bins=25, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(errors_vox.mean(), color="red", ls="--", lw=2,
               label=f"Mean: {errors_vox.mean():.1f} vox")
    ax.axvline(np.median(errors_vox), color="orange", ls="--", lw=2,
               label=f"Median: {np.median(errors_vox):.1f} vox")
    ax.set_xlabel("Localization Error (voxels)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Error Distribution (Voxel)", fontsize=12)
    ax.legend(fontsize=9)
    ax2 = ax.twiny()
    ax2.hist(errors_mm, bins=25, alpha=0)
    ax2.set_xlabel("(mm)", fontsize=9, color="gray")
    ax2.tick_params(axis="x", labelsize=8, colors="gray")

    ax = fig.add_subplot(gs[0, 1])
    sorted_mm = np.sort(errors_mm)
    cdf = np.arange(1, len(sorted_mm) + 1) / len(sorted_mm) * 100
    ax.plot(sorted_mm, cdf, "b-", linewidth=2)
    ax.fill_between(sorted_mm, 0, cdf, alpha=0.15, color="blue")
    for thr_mm, color in [(2.0, "green"), (5.0, "orange"), (10.0, "red")]:
        pct = (errors_mm < thr_mm).mean() * 100
        ax.axvline(thr_mm, color=color, ls=":", lw=1.5, alpha=0.7)
        ax.annotate(
            f"<{thr_mm}mm: {pct:.0f}%",
            xy=(thr_mm, pct),
            fontsize=9,
            xytext=(thr_mm + 0.3, max(pct - 8, 5)),
            color=color,
            fontweight="bold",
        )
    ax.set_xlabel("Error (mm)", fontsize=11)
    ax.set_ylabel("Cumulative %", fontsize=11)
    ax.set_title("CDF (Physical Distance)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, :], projection="3d")
    sc = ax.scatter(
        offsets[:, 0],
        offsets[:, 1],
        offsets[:, 2],
        c=errors_mm,
        cmap="RdYlGn_r",
        s=30,
        alpha=0.7,
        edgecolors="gray",
        linewidths=0.3,
    )
    ax.scatter([0], [0], [0], c="black", s=100, marker="x", linewidths=2,
               label="Origin (GT)")
    ax.set_xlabel("Dim0 offset (vox)", fontsize=9)
    ax.set_ylabel("Dim1 offset (vox)", fontsize=9)
    ax.set_zlabel("Dim2 offset (vox)", fontsize=9)
    ax.set_title("Prediction Offset in 3D", fontsize=12)
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, label="Error (mm)", shrink=0.55, pad=0.02)

    p = os.path.join(out_dir, "verification_dashboard_compact.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {p}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=str,
        default="verify_results_dual_head_100ep/verification_results.csv",
    )
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--prepared_dir", type=str, default=None)
    p.add_argument("--ckpt_name", type=str, default="dual_head_100ep")
    p.add_argument(
        "--example_dim",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="第二行三例使用的同一解剖维度: 0=矢状 1=冠状 2=轴位(默认)",
    )
    p.add_argument("--legacy_compact", action="store_true",
                   help="同时输出旧版 compact (上2+下3D)")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)

    prepared_dir = args.prepared_dir
    if prepared_dir is None:
        try:
            import yaml

            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            prepared_dir = cfg["data"]["prepared_dir"]
        except Exception:
            with open(args.config) as f:
                for line in f:
                    if "prepared_dir:" in line and not line.strip().startswith("#"):
                        prepared_dir = line.split(":", 1)[1].split("#")[0].strip()
                        break
            if not prepared_dir:
                prepared_dir = os.path.join(REPO_ROOT, 'prepared_data')

    results = load_results(args.csv)
    if not results:
        raise SystemExit(f"No rows in {args.csv}")

    out_png = os.path.join(out_dir, "verification_dashboard_2x3.png")
    plot_dashboard_2x3(
        results,
        prepared_dir,
        out_png,
        ckpt_name=args.ckpt_name,
        example_dim=args.example_dim,
    )

    if args.legacy_compact:
        plot_top2_and_3d_legacy(results, out_dir, ckpt_name=args.ckpt_name)


if __name__ == "__main__":
    main()
