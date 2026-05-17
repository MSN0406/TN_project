import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def dice2d(pred_slice: np.ndarray, gt_slice: np.ndarray, cls: int) -> float:
    p = pred_slice == cls
    g = gt_slice == cls
    den = p.sum() + g.sum()
    if den == 0:
        return float("nan")
    return float(2.0 * (p & g).sum() / den)


def get_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return vol[idx, :, :]
    if axis == 1:
        return vol[:, idx, :]
    return vol[:, :, idx]


def make_label_rgb(label: np.ndarray, roi_slice: np.ndarray) -> np.ndarray:
    if (roi_slice > 0).any():
        vmin, vmax = np.percentile(roi_slice[roi_slice > 0], [1, 99])
    else:
        vmin, vmax = 0.0, 1.0
    gray = np.clip((roi_slice - vmin) / (vmax - vmin + 1e-8), 0, 1)
    rgb = np.stack([gray, gray, gray], axis=-1)

    nerve = label == 1
    vessel = label == 2
    if nerve.any():
        rgb[nerve] = rgb[nerve] * 0.3 + np.array([1.0, 0.2, 0.2]) * 0.7
    if vessel.any():
        rgb[vessel] = rgb[vessel] * 0.3 + np.array([0.3, 0.5, 1.0]) * 0.7
    return rgb


def make_tp_fp_fn_overlay(gt_slice: np.ndarray, pred_slice: np.ndarray, roi_slice: np.ndarray, num_classes=3):
    if (roi_slice > 0).any():
        vmin, vmax = np.percentile(roi_slice[roi_slice > 0], [1, 99])
    else:
        vmin, vmax = 0.0, 1.0
    gray = np.clip((roi_slice - vmin) / (vmax - vmin + 1e-8), 0, 1)
    diff = np.stack([gray, gray, gray], axis=-1) * 0.4

    for c in range(1, num_classes):
        tp = (pred_slice == c) & (gt_slice == c)
        fp = (pred_slice == c) & (gt_slice != c)
        fn = (pred_slice != c) & (gt_slice == c)
        diff[tp] = [0.0, 1.0, 0.0]   # TP green
        diff[fp] = [1.0, 1.0, 0.0]   # FP yellow
        diff[fn] = [0.8, 0.0, 0.8]   # FN purple
    return diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True, type=str)
    parser.add_argument("--gt_dir", required=True, type=str)
    parser.add_argument("--img_dir", type=str, default=None,
                        help="原始图像目录(imagesTr). 若不提供，默认用 gt_dir 的兄弟目录 imagesTr")
    parser.add_argument("--splits_file", required=True, type=str)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, default="random", choices=["random", "all"],
                        help="random: 随机抽样 num_cases 个; all: 使用该 fold 的全部 val cases")
    parser.add_argument("--cases_from_dir", type=str, default=None,
                        help="从参考目录读取 case 列表（根据 *_best.png / *_worst.png 文件名提取 case_id）")
    parser.add_argument("--top_k", type=int, default=5,
                        help="汇总输出里导出全局 best/worst case 数量")
    parser.add_argument("--output_dir", required=True, type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    img_dir = args.img_dir if args.img_dir else os.path.join(os.path.dirname(args.gt_dir), "imagesTr")

    with open(args.splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    val_cases = splits[args.fold]["val"]

    rng = random.Random(args.seed)
    if args.cases_from_dir:
        ref_dir = Path(args.cases_from_dir)
        case_ids = set()
        for p in ref_dir.glob("*.png"):
            name = p.name
            if name.endswith("_best.png"):
                case_ids.add(name[:-9])
            elif name.endswith("_worst.png"):
                case_ids.add(name[:-10])
        cases = [cid for cid in sorted(case_ids) if cid in set(val_cases)]
        print(f"[info] cases_from_dir: {args.cases_from_dir}")
        print(f"[info] extracted {len(case_ids)} cases, matched val cases: {len(cases)}")
        if not cases:
            raise RuntimeError("cases_from_dir 未匹配到任何 val case，请检查目录与 fold 是否对应")
    elif args.mode == "all":
        cases = list(val_cases)
    else:
        cases = rng.sample(val_cases, min(args.num_cases, len(val_cases)))

    summary = []
    avg_nerve_list, avg_vessel_list = [], []
    class_names = {1: "Nerve", 2: "Vessel"}
    axis_names = ["Dim0 (Sagittal)", "Dim1 (Coronal)", "Dim2 (Axial)"]
    colors = {1: "#FF4444", 2: "#4488FF"}

    for cid in cases:
        gt_path = os.path.join(args.gt_dir, f"{cid}.nii.gz")
        pred_path = os.path.join(args.pred_dir, f"{cid}.nii.gz")
        img_path = os.path.join(img_dir, f"{cid}_0000.nii.gz")
        if not os.path.exists(pred_path):
            print(f"[skip] missing pred: {pred_path}")
            continue
        if not os.path.exists(img_path):
            print(f"[skip] missing image: {img_path}")
            continue

        gt = nib.load(gt_path).get_fdata().astype(np.int16)
        pred = nib.load(pred_path).get_fdata().astype(np.int16)
        img = nib.load(img_path).get_fdata().astype(np.float32)

        best = {"axis": 0, "slice": 0, "mean": -1.0, "nerve": float("nan"), "vessel": float("nan")}
        worst = {"axis": 0, "slice": 0, "mean": 2.0, "nerve": float("nan"), "vessel": float("nan")}
        nerve_dices, vessel_dices = [], []
        all_dices = {0: {1: [float("nan")] * gt.shape[0], 2: [float("nan")] * gt.shape[0]},
                     1: {1: [float("nan")] * gt.shape[1], 2: [float("nan")] * gt.shape[1]},
                     2: {1: [float("nan")] * gt.shape[2], 2: [float("nan")] * gt.shape[2]}}

        for ax in range(3):
            for s in range(gt.shape[ax]):
                g = get_slice(gt, ax, s)
                p = get_slice(pred, ax, s)
                has_nerve = np.any(g == 1)
                has_vessel = np.any(g == 2)
                if not (has_nerve or has_vessel):
                    continue

                d_nerve = dice2d(p, g, 1) if has_nerve else float("nan")
                d_vessel = dice2d(p, g, 2) if has_vessel else float("nan")
                all_dices[ax][1][s] = d_nerve
                all_dices[ax][2][s] = d_vessel
                if has_nerve:
                    nerve_dices.append(d_nerve)
                if has_vessel:
                    vessel_dices.append(d_vessel)

                valid = [d for d in (d_nerve, d_vessel) if not np.isnan(d)]
                m = float(np.mean(valid)) if valid else float("nan")
                if not np.isnan(m) and m > best["mean"]:
                    best = {"axis": ax, "slice": s, "mean": m, "nerve": d_nerve, "vessel": d_vessel}
                if not np.isnan(m) and m < worst["mean"]:
                    worst = {"axis": ax, "slice": s, "mean": m, "nerve": d_nerve, "vessel": d_vessel}

        avg_nerve = float(np.nanmean(nerve_dices)) if nerve_dices else float("nan")
        avg_vessel = float(np.nanmean(vessel_dices)) if vessel_dices else float("nan")
        avg_nerve_list.append(avg_nerve)
        avg_vessel_list.append(avg_vessel)

        for tag, sel in [("best", best), ("worst", worst)]:
            g = get_slice(gt, sel["axis"], sel["slice"])
            p = get_slice(pred, sel["axis"], sel["slice"])
            r = get_slice(img, sel["axis"], sel["slice"])
            cmp_rgb = make_tp_fp_fn_overlay(g, p, r, num_classes=3)

            fig = plt.figure(figsize=(20, 12))
            gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)

            fig.suptitle(
                f"{cid}  |  {tag.title()} slice: {axis_names[sel['axis']]} #{sel['slice']}  |  "
                f"Mean Dice={sel['mean']:.4f}  |  "
                f"Nerve={sel['nerve']:.4f}, Vessel={sel['vessel']:.4f}",
                fontsize=13, y=0.98,
            )

            titles = ["ROI", "GT", "Prediction", "Comparison (G=TP, Y=FP, P=FN)"]
            images = [
                r.T,
                make_label_rgb(g.T, r.T),
                make_label_rgb(p.T, r.T),
                cmp_rgb.transpose(1, 0, 2),
            ]

            for col in range(4):
                ax = fig.add_subplot(gs[0, col])
                if col == 0:
                    ax.imshow(images[col], cmap="gray", origin="lower")
                else:
                    ax.imshow(images[col], origin="lower")
                ax.set_title(titles[col], fontsize=11)
                ax.axis("off")

            for plot_axis in range(3):
                ax = fig.add_subplot(gs[1, plot_axis])
                for c in (1, 2):
                    dices = all_dices[plot_axis][c]
                    valid_x = [i for i, d in enumerate(dices) if not np.isnan(d)]
                    valid_d = [d for d in dices if not np.isnan(d)]
                    ax.plot(valid_x, valid_d, color=colors[c], linewidth=1.2,
                            label=class_names[c], alpha=0.8)
                if plot_axis == best["axis"]:
                    ax.axvline(best["slice"], color="lime", linewidth=2, linestyle="--",
                               label=f"Best ({best['slice']})", alpha=0.8)
                if plot_axis == worst["axis"]:
                    ax.axvline(worst["slice"], color="red", linewidth=2, linestyle="--",
                               label=f"Worst ({worst['slice']})", alpha=0.8)
                ax.set_title(axis_names[plot_axis], fontsize=10)
                ax.set_xlabel("Slice index")
                ax.set_ylabel("2D Dice")
                ax.set_ylim(-0.05, 1.05)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            ax = fig.add_subplot(gs[1, 3])
            ax.axis("off")
            summary_text = (
                f"Case: {cid}\n\n"
                f"Best axis/slice: {axis_names[best['axis']]} / {best['slice']}\n"
                f"Best mean: {best['mean']:.4f}\n"
                f"Nerve(best): {best['nerve']:.4f}\n"
                f"Vessel(best): {best['vessel']:.4f}\n\n"
                f"Worst axis/slice: {axis_names[worst['axis']]} / {worst['slice']}\n"
                f"Worst mean: {worst['mean']:.4f}\n"
                f"Nerve(worst): {worst['nerve']:.4f}\n"
                f"Vessel(worst): {worst['vessel']:.4f}\n\n"
                f"Nerve(avg): {avg_nerve:.4f}\n"
                f"Vessel(avg): {avg_vessel:.4f}\n"
            )
            ax.text(
                0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            )

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(args.output_dir, f"{cid}_{tag}.png"), dpi=150)
            plt.close(fig)

        summary.append(
            {
                "case_id": cid,
                "best_axis": int(best["axis"]),
                "best_slice": int(best["slice"]),
                "best_mean_2d_dice": float(best["mean"]),
                "best_nerve_2d_dice": None if np.isnan(best["nerve"]) else float(best["nerve"]),
                "best_vessel_2d_dice": None if np.isnan(best["vessel"]) else float(best["vessel"]),
                "worst_axis": int(worst["axis"]),
                "worst_slice": int(worst["slice"]),
                "worst_mean_2d_dice": float(worst["mean"]),
                "worst_nerve_2d_dice": None if np.isnan(worst["nerve"]) else float(worst["nerve"]),
                "worst_vessel_2d_dice": None if np.isnan(worst["vessel"]) else float(worst["vessel"]),
                "avg_nerve_2d_dice": None if np.isnan(avg_nerve) else avg_nerve,
                "avg_vessel_2d_dice": None if np.isnan(avg_vessel) else avg_vessel,
            }
        )

    out_json = os.path.join(args.output_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    valid_summary = [
        s for s in summary
        if s["avg_nerve_2d_dice"] is not None and s["avg_vessel_2d_dice"] is not None
    ]
    for s in valid_summary:
        s["_avg_mean_2d_dice"] = (s["avg_nerve_2d_dice"] + s["avg_vessel_2d_dice"]) / 2.0

    valid_summary.sort(key=lambda x: x["_avg_mean_2d_dice"], reverse=True)
    k = min(args.top_k, len(valid_summary))
    ranking = {
        "top_k_cases": [
            {
                "case_id": s["case_id"],
                "avg_mean_2d_dice": s["_avg_mean_2d_dice"],
                "avg_nerve_2d_dice": s["avg_nerve_2d_dice"],
                "avg_vessel_2d_dice": s["avg_vessel_2d_dice"],
            }
            for s in valid_summary[:k]
        ],
        "bottom_k_cases": [
            {
                "case_id": s["case_id"],
                "avg_mean_2d_dice": s["_avg_mean_2d_dice"],
                "avg_nerve_2d_dice": s["avg_nerve_2d_dice"],
                "avg_vessel_2d_dice": s["avg_vessel_2d_dice"],
            }
            for s in valid_summary[-k:]
        ],
    }
    with open(os.path.join(args.output_dir, "ranking_top_bottom.json"), "w", encoding="utf-8") as f:
        json.dump(ranking, f, indent=2, ensure_ascii=False)

    print(f"[done] output: {args.output_dir}")
    if avg_nerve_list:
        print(f"[done] mean(avg_nerve_2d_dice): {np.nanmean(avg_nerve_list):.4f}")
    if avg_vessel_list:
        print(f"[done] mean(avg_vessel_2d_dice): {np.nanmean(avg_vessel_list):.4f}")


if __name__ == "__main__":
    main()

