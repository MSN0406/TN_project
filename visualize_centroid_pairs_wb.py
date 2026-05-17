"""
visualize_centroid_pairs_wb.py
==============================
在 whole brain 上同时标注同一 MRN 的 ipsi / contra centroid。

输出:
  - 每个 MRN 一张图
  - 3 行 x 2 列:
      每行对应一个轴(dim0/dim1/dim2)
      左列切 ipsi 对应 slice, 右列切 contra 对应 slice
  - 每个子图都同时画出 ipsi(绿色 x) 与 contra(橙色 +) 的投影位置
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import yaml


def load_case(prepared_dir, case_id):
    case_dir = os.path.join(prepared_dir, case_id)
    with open(os.path.join(case_dir, "info.json")) as f:
        info = json.load(f)
    centroid = np.load(os.path.join(case_dir, "centroid.npy")).astype(int)
    return info, centroid


def pick_mrn_pairs(prepared_dir):
    with open(os.path.join(prepared_dir, "metadata.json")) as f:
        case_ids = json.load(f)["case_ids"]

    by_mrn = {}
    for cid in case_ids:
        mrn, side = cid.rsplit("_", 1)
        by_mrn.setdefault(mrn, set()).add(side)
    return sorted([m for m, sides in by_mrn.items() if "ipsi" in sides and "contra" in sides])


def get_slice_and_points(wb, ipsi_c, contra_c, axis, use_side):
    idx = int(ipsi_c[axis] if use_side == "ipsi" else contra_c[axis])
    idx = int(np.clip(idx, 0, wb.shape[axis] - 1))

    if axis == 0:
        sl = wb[idx, :, :].T
        ipsi_xy = (ipsi_c[1], ipsi_c[2])
        contra_xy = (contra_c[1], contra_c[2])
    elif axis == 1:
        sl = wb[:, idx, :].T
        ipsi_xy = (ipsi_c[0], ipsi_c[2])
        contra_xy = (contra_c[0], contra_c[2])
    else:
        sl = wb[:, :, idx].T
        ipsi_xy = (ipsi_c[0], ipsi_c[1])
        contra_xy = (contra_c[0], contra_c[1])
    return sl, idx, ipsi_xy, contra_xy


def vis_one_mrn(prepared_dir, mrn, output_dir):
    ipsi_id = f"{mrn}_ipsi"
    contra_id = f"{mrn}_contra"

    info_i, c_i = load_case(prepared_dir, ipsi_id)
    info_c, c_c = load_case(prepared_dir, contra_id)

    nii_i = info_i["nii_path"]
    nii_c = info_c["nii_path"]
    if nii_i != nii_c:
        raise RuntimeError(f"{mrn}: ipsi/contra 对应的 NIfTI 不一致")

    wb = nib.load(nii_i).get_fdata().astype(np.float32)
    if wb.ndim == 4:
        wb = wb[..., 0]

    axis_names = ["Dim0 (Sagittal)", "Dim1 (Coronal)", "Dim2 (Axial)"]
    fig, axes = plt.subplots(3, 2, figsize=(13, 16))
    fig.suptitle(
        f"MRN {mrn} | ipsi={c_i.tolist()} | contra={c_c.tolist()}",
        fontsize=13, y=0.98
    )

    for ax_idx in range(3):
        for col, side in enumerate(["ipsi", "contra"]):
            sl, idx, ipsi_xy, contra_xy = get_slice_and_points(wb, c_i, c_c, ax_idx, side)
            vmin, vmax = np.percentile(sl[sl > 0], [1, 99]) if (sl > 0).any() else (0, 1)
            plot_ax = axes[ax_idx, col]
            plot_ax.imshow(sl, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
            plot_ax.plot(ipsi_xy[0], ipsi_xy[1], "x", color="lime", markersize=12, markeredgewidth=2, label="ipsi")
            plot_ax.plot(contra_xy[0], contra_xy[1], "+", color="orange", markersize=12, markeredgewidth=2, label="contra")
            plot_ax.set_title(f"{axis_names[ax_idx]} @ {side} slice={idx}", fontsize=10)
            if ax_idx == 0 and col == 1:
                plot_ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(output_dir, f"{mrn}_ipsi_contra_centroid.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num_cases", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="viz_centroid_pairs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    prepared_dir = cfg["data"]["prepared_dir"]
    os.makedirs(args.output_dir, exist_ok=True)

    mrns = pick_mrn_pairs(prepared_dir)
    rng = np.random.RandomState(args.seed)
    chosen = sorted(rng.choice(mrns, size=min(args.num_cases, len(mrns)), replace=False).tolist())

    print(f"可视化 {len(chosen)} 个 MRN（ipsi+contra 同图）:")
    for mrn in chosen:
        out = vis_one_mrn(prepared_dir, mrn, args.output_dir)
        print(f"  saved: {out}")


if __name__ == "__main__":
    main()
