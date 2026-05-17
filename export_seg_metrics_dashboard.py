#!/usr/bin/env python3
"""
从 final_results.csv（verify_final.py 导出）生成分割指标图:
  - 默认输出 seg_metrics_top2.png（仅前两个子图: Nerve | Vessel 直方图, 1×2）
  - 同时输出 seg_metrics_dashboard_2x3.png（完整 6 格）；若只要前两个图可加 --top2_only

示例:
  python export_seg_metrics_dashboard.py --csv verify_final_results/final_results.csv --top2_only
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def plot_seg_top2(rows, out_path, title_tag=""):
    """仅前两个子图: Nerve Dice | Vessel Dice 直方图 (1×2)."""
    dn = np.array([r["dice_nerve"] for r in rows])
    dv = np.array([r["dice_vessel"] for r in rows])
    em = np.array([r["error_mm"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tag = f" | {title_tag}" if title_tag else ""
    fig.suptitle(
        f"Segmentation Metrics (top-2){tag}  |  n={len(rows)} cases\n"
        f"Mean Dice Nerve: {np.mean(dn):.4f}  |  Mean Dice Vessel: {np.mean(dv):.4f}  |  "
        f"Mean Loc err: {np.mean(em):.2f} mm",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0]
    ax.hist(dn, bins=22, color="#c0392b", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(dn), color="navy", ls="--", lw=2, label=f"Mean: {np.mean(dn):.4f}")
    ax.axvline(np.median(dn), color="darkorange", ls="--", lw=2,
               label=f"Median: {np.median(dn):.4f}")
    ax.set_xlabel("Dice (nerve)")
    ax.set_ylabel("Count")
    ax.set_title("Nerve Dice (2D slice-wise, 48³ region)")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(dv, bins=22, color="#2980b9", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(dv), color="navy", ls="--", lw=2, label=f"Mean: {np.mean(dv):.4f}")
    ax.axvline(np.median(dv), color="darkorange", ls="--", lw=2,
               label=f"Median: {np.median(dv):.4f}")
    ax.set_xlabel("Dice (vessel)")
    ax.set_ylabel("Count")
    ax.set_title("Vessel Dice (2D slice-wise, 48³ region)")
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def load_final_csv(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "case_id": row["case_id"],
                    "side": row["side"],
                    "error_mm": float(row["error_mm"]),
                    "dice_nerve": float(row["dice_nerve"]),
                    "dice_vessel": float(row["dice_vessel"]),
                }
            )
    return rows


def plot_seg_dashboard(rows, out_path, title_tag=""):
    dn = np.array([r["dice_nerve"] for r in rows])
    dv = np.array([r["dice_vessel"] for r in rows])
    em = np.array([r["error_mm"] for r in rows])

    n_ipsi = [r["dice_nerve"] for r in rows if r["side"] == "ipsi"]
    n_contra = [r["dice_nerve"] for r in rows if r["side"] == "contra"]
    v_ipsi = [r["dice_vessel"] for r in rows if r["side"] == "ipsi"]
    v_contra = [r["dice_vessel"] for r in rows if r["side"] == "contra"]

    fig = plt.figure(figsize=(20, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.26)
    tag = f" | {title_tag}" if title_tag else ""
    fig.suptitle(
        f"Segmentation Metrics (2×3){tag}  |  n={len(rows)} cases\n"
        f"Mean Dice Nerve: {np.mean(dn):.4f}  |  Mean Dice Vessel: {np.mean(dv):.4f}  |  "
        f"Mean Loc err: {np.mean(em):.2f} mm",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # (0,0) Nerve dice hist
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(dn, bins=22, color="#c0392b", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(dn), color="navy", ls="--", lw=2, label=f"Mean: {np.mean(dn):.4f}")
    ax.axvline(np.median(dn), color="darkorange", ls="--", lw=2,
               label=f"Median: {np.median(dn):.4f}")
    ax.set_xlabel("Dice (nerve)")
    ax.set_ylabel("Count")
    ax.set_title("Nerve Dice (2D slice-wise, 48³ region)")
    ax.legend(fontsize=9)

    # (0,1) Vessel dice hist
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(dv, bins=22, color="#2980b9", edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(dv), color="navy", ls="--", lw=2, label=f"Mean: {np.mean(dv):.4f}")
    ax.axvline(np.median(dv), color="darkorange", ls="--", lw=2,
               label=f"Median: {np.median(dv):.4f}")
    ax.set_xlabel("Dice (vessel)")
    ax.set_ylabel("Count")
    ax.set_title("Vessel Dice (2D slice-wise, 48³ region)")
    ax.legend(fontsize=9)

    # (0,2) Ipsi vs Contra — nerve dice
    ax = fig.add_subplot(gs[0, 2])
    data_box, labels_box = [], []
    if n_ipsi:
        data_box.append(n_ipsi)
        labels_box.append(f"Ipsi (n={len(n_ipsi)})\nμ={np.mean(n_ipsi):.4f}")
    if n_contra:
        data_box.append(n_contra)
        labels_box.append(f"Contra (n={len(n_contra)})\nμ={np.mean(n_contra):.4f}")
    if data_box:
        bp = ax.boxplot(data_box, tick_labels=labels_box, patch_artist=True, widths=0.5)
        colors_box = ["#3498db", "#e74c3c"]
        for patch, c in zip(bp["boxes"], colors_box[: len(data_box)]):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
    ax.set_ylabel("Dice (nerve)")
    ax.set_title("Nerve Dice: Ipsi vs Contra")
    ax.grid(True, alpha=0.3, axis="y")

    # (1,0) Nerve vs Vessel
    ax = fig.add_subplot(gs[1, 0])
    sides = np.array([r["side"] for r in rows])
    c = np.where(sides == "ipsi", "#3498db", "#e74c3c")
    ax.scatter(dn, dv, c=c, s=36, alpha=0.75, edgecolors="gray", linewidths=0.4)
    ax.set_xlabel("Dice nerve")
    ax.set_ylabel("Dice vessel")
    ax.set_title("Nerve vs Vessel (per case)")
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D

    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
               markersize=8, label="ipsi"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=8, label="contra"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=9)

    # (1,1) CDF nerve dice
    ax = fig.add_subplot(gs[1, 1])
    sdn = np.sort(dn)
    cdf = np.arange(1, len(sdn) + 1) / len(sdn) * 100
    ax.plot(sdn, cdf, color="#c0392b", lw=2)
    ax.fill_between(sdn, 0, cdf, alpha=0.12, color="#c0392b")
    for thr, color in [(0.5, "green"), (0.6, "orange"), (0.7, "blue")]:
        pct = (dn >= thr).mean() * 100
        ax.axvline(thr, color=color, ls=":", lw=1.5, alpha=0.7)
        ax.annotate(
            f">={thr:.1f}: {pct:.0f}%",
            xy=(thr, pct),
            fontsize=9,
            xytext=(thr + 0.02, max(pct - 10, 8)),
            color=color,
            fontweight="bold",
        )
    ax.set_xlabel("Dice (nerve)")
    ax.set_ylabel("Cumulative %")
    ax.set_title("CDF (Nerve Dice)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # (1,2) Loc error vs Nerve dice
    ax = fig.add_subplot(gs[1, 2])
    sc = ax.scatter(em, dn, c=dv, cmap="viridis", s=38, alpha=0.8,
                    edgecolors="gray", linewidths=0.4)
    plt.colorbar(sc, ax=ax, label="Dice vessel")
    ax.set_xlabel("Localization error (mm)")
    ax.set_ylabel("Dice nerve")
    ax.set_title("Loc error vs Seg (nerve); color=vessel Dice")
    ax.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="verify_final_results/final_results.csv")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--out_name", type=str, default="seg_metrics_dashboard_2x3.png")
    p.add_argument(
        "--top2_name",
        type=str,
        default="seg_metrics_top2.png",
        help="仅前两个子图时的输出文件名",
    )
    p.add_argument(
        "--top2_only",
        action="store_true",
        help="只生成前两个子图 (1×2). 不生成完整 2×3",
    )
    p.add_argument("--title_tag", type=str, default="")
    args = p.parse_args()
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)
    rows = load_final_csv(args.csv)
    if not rows:
        raise SystemExit(f"No rows in {args.csv}")

    top2_path = os.path.join(out_dir, args.top2_name)
    plot_seg_top2(rows, top2_path, title_tag=args.title_tag)

    if not args.top2_only:
        out_path = os.path.join(out_dir, args.out_name)
        plot_seg_dashboard(rows, out_path, title_tag=args.title_tag)


if __name__ == "__main__":
    main()
