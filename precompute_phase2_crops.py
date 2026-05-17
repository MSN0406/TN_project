import os
REPO_ROOT = os.environ.get('TN_ROOT', os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import os
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from nnunet_loc_trainer import nnUNetTrainerLoc


def _to_tuple3(values: List[int]) -> Tuple[int, int, int]:
    return int(values[0]), int(values[1]), int(values[2])


def zscore_wb(wb: np.ndarray) -> np.ndarray:
    fg = wb > 0
    if fg.sum() > 100:
        mean = float(wb[fg].mean())
        std = float(wb[fg].std())
    else:
        mean = float(wb.mean())
        std = float(wb.std())
    return ((wb - mean) / (std + 1e-8)).astype(np.float32, copy=False)


def crop_with_padding(vol: np.ndarray, center_dhw: np.ndarray, crop_size: Tuple[int, int, int], fill_value=0):
    d, h, w = vol.shape[:3]
    td, th, tw = crop_size
    hd, hh, hw = td // 2, th // 2, tw // 2

    c = np.round(center_dhw).astype(int)
    start = np.array([c[0] - hd, c[1] - hh, c[2] - hw], dtype=int)
    end = start + np.array([td, th, tw], dtype=int)

    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, np.array([d, h, w], dtype=int))
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    out = np.full((td, th, tw), fill_value=fill_value, dtype=vol.dtype)
    out[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = vol[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]
    return out


def crop_pair_with_padding(
    img: np.ndarray,
    lbl: np.ndarray,
    center_dhw: np.ndarray,
    crop_size: Tuple[int, int, int],
):
    """
    使用完全相同的起止索引同时裁剪 image 和 label，保证像素级对齐。
    """
    d, h, w = img.shape[:3]
    td, th, tw = crop_size
    hd, hh, hw = td // 2, th // 2, tw // 2

    c = np.round(center_dhw).astype(int)
    start = np.array([c[0] - hd, c[1] - hh, c[2] - hw], dtype=int)
    end = start + np.array([td, th, tw], dtype=int)

    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, np.array([d, h, w], dtype=int))
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    out_img = np.zeros((td, th, tw), dtype=img.dtype)
    out_lbl = np.zeros((td, th, tw), dtype=lbl.dtype)
    out_img[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = img[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]
    out_lbl[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = lbl[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]
    return out_img, out_lbl, start, end


def place_mask_in_wb(wb_shape: Tuple[int, int, int], gt_centroid_vox: np.ndarray, mask_crop: np.ndarray) -> np.ndarray:
    mask_wb = np.zeros(wb_shape, dtype=np.uint8)
    crop_shape = np.array(mask_crop.shape, dtype=int)
    center = np.round(gt_centroid_vox).astype(int)

    starts = center - crop_shape // 2
    ends = starts + crop_shape

    wb_starts = np.maximum(starts, 0)
    wb_ends = np.minimum(ends, np.array(wb_shape, dtype=int))
    crop_starts = wb_starts - starts
    crop_ends = crop_shape - (ends - wb_ends)

    mask_wb[
        wb_starts[0]:wb_ends[0],
        wb_starts[1]:wb_ends[1],
        wb_starts[2]:wb_ends[2],
    ] = mask_crop[
        crop_starts[0]:crop_ends[0],
        crop_starts[1]:crop_ends[1],
        crop_starts[2]:crop_ends[2],
    ]
    return mask_wb


def mask_foreground_centroid(mask_wb: np.ndarray, fallback_center: np.ndarray) -> np.ndarray:
    fg = np.argwhere(mask_wb > 0)
    if fg.shape[0] == 0:
        return fallback_center.astype(np.float32, copy=False)
    return fg.mean(axis=0).astype(np.float32, copy=False)


def load_phase1_localizer(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "network_weights" not in ckpt:
        raise RuntimeError("checkpoint is not nnUNetTrainerLoc style (missing network_weights)")
    init_args = ckpt["init_args"]
    trainer = nnUNetTrainerLoc(
        plans=init_args["plans"],
        configuration=init_args["configuration"],
        fold=init_args["fold"],
        dataset_json=init_args["dataset_json"],
        device=device,
    )
    trainer.initialize()
    trainer.network.load_state_dict(ckpt["network_weights"])
    trainer.network.eval()
    return trainer


@torch.no_grad()
def predict_centroid_norm(trainer: nnUNetTrainerLoc, wb_norm: np.ndarray, gt_centroid_norm: np.ndarray) -> np.ndarray:
    wb_t = torch.from_numpy(wb_norm).unsqueeze(0).unsqueeze(0).to(trainer.device)
    mod = trainer.network.module if trainer.is_ddp else trainer.network

    loc_in = wb_t
    if tuple(wb_t.shape[2:]) != tuple(trainer.loc_input_size):
        loc_in = F.interpolate(wb_t, size=tuple(trainer.loc_input_size), mode="trilinear", align_corners=False)

    hm_l, hm_r, c_l, c_r = mod.forward_loc(loc_in)
    use_left = bool(float(gt_centroid_norm[0]) < 0.5)
    c = c_l if use_left else c_r
    return c.squeeze(0).detach().cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_dir", type=str, default=os.path.join(REPO_ROOT, 'prepared_data'))
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(REPO_ROOT, 'nnUNet_data/nnUNet_results/Dataset001_TN/nnUNetTrainerLoc__nnUNetPlans__3d_fullres/fold_0/phase1_best.pth'),
    )
    parser.add_argument("--output_dir", type=str, default=os.path.join(REPO_ROOT, 'precomputed_phase2_crop96_pred'))
    parser.add_argument("--crop_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=0, help="0 means all cases")
    parser.add_argument(
        "--center_mode",
        type=str,
        default="pred",
        choices=["pred", "gt", "mix", "mask_centroid"],
        help="crop 中心选择: pred=phase1预测, gt=GT中心, mix=二者线性插值, mask_centroid=mask前景质心",
    )
    parser.add_argument(
        "--mix_alpha",
        type=float,
        default=0.5,
        help="当 center_mode=mix 时, center=(1-alpha)*gt + alpha*pred",
    )
    parser.add_argument(
        "--max_shift_vox",
        type=float,
        default=0.0,
        help="限制相对 GT 的最大偏移(体素). 0 表示不限制",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    crop_size = _to_tuple3(args.crop_size)

    device = torch.device(type="cuda", index=args.gpu) if torch.cuda.is_available() else torch.device("cpu")
    print(f"[Info] device={device}, crop_size={crop_size}")
    need_localizer = args.center_mode in ("pred", "mix")
    trainer = load_phase1_localizer(args.checkpoint, device) if need_localizer else None

    meta_path = os.path.join(args.prepared_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"metadata not found: {meta_path}")
    with open(meta_path) as f:
        metadata = json.load(f)
    case_ids = list(metadata["case_ids"])
    if args.max_cases > 0:
        case_ids = case_ids[: args.max_cases]

    stats = []
    for case_id in tqdm(case_ids, desc="Precompute crops", dynamic_ncols=True):
        case_dir = os.path.join(args.prepared_dir, case_id)
        out_case_dir = os.path.join(args.output_dir, case_id)
        os.makedirs(out_case_dir, exist_ok=True)

        with open(os.path.join(case_dir, "info.json")) as f:
            info = json.load(f)
        wb = nib.load(info["nii_path"]).get_fdata().astype(np.float32)
        if wb.ndim == 4:
            wb = wb[..., 0]
        wb_norm = zscore_wb(wb)

        gt_centroid_vox = np.load(os.path.join(case_dir, "centroid.npy")).astype(np.float32)
        wb_shape = np.array(wb.shape[:3], dtype=np.float32)
        gt_centroid_norm = gt_centroid_vox / np.maximum(wb_shape - 1.0, 1.0)

        mask_crop = nib.load(os.path.join(case_dir, "mask_aligned.nii.gz")).get_fdata().astype(np.uint8)
        mask_wb = place_mask_in_wb(tuple(int(i) for i in wb.shape[:3]), gt_centroid_vox, mask_crop)
        mask_centroid_vox = mask_foreground_centroid(mask_wb, gt_centroid_vox)

        if need_localizer:
            pred_centroid_norm = predict_centroid_norm(trainer, wb_norm, gt_centroid_norm)
            pred_centroid_vox = pred_centroid_norm * (wb_shape - 1.0)
            loc_err = float(np.linalg.norm(pred_centroid_vox - gt_centroid_vox))
        else:
            pred_centroid_vox = gt_centroid_vox.copy()
            loc_err = float(np.linalg.norm(pred_centroid_vox - gt_centroid_vox))

        if args.center_mode == "gt":
            center_vox = gt_centroid_vox.copy()
        elif args.center_mode == "mix":
            alpha = float(np.clip(args.mix_alpha, 0.0, 1.0))
            center_vox = (1.0 - alpha) * gt_centroid_vox + alpha * pred_centroid_vox
        elif args.center_mode == "mask_centroid":
            center_vox = mask_centroid_vox.copy()
        else:
            center_vox = pred_centroid_vox.copy()

        if args.max_shift_vox > 0 and args.center_mode in ("pred", "mix"):
            delta = center_vox - gt_centroid_vox
            dist = float(np.linalg.norm(delta))
            if dist > args.max_shift_vox and dist > 1e-6:
                center_vox = gt_centroid_vox + delta / dist * float(args.max_shift_vox)

        img_crop, lbl_crop, crop_start, crop_end = crop_pair_with_padding(
            wb_norm, mask_wb, center_vox, crop_size
        )
        img_crop = img_crop.astype(np.float32, copy=False)
        lbl_crop = lbl_crop.astype(np.uint8, copy=False)

        np.save(os.path.join(out_case_dir, "image.npy"), img_crop)
        np.save(os.path.join(out_case_dir, "label.npy"), lbl_crop)
        np.save(os.path.join(out_case_dir, "pred_centroid_vox.npy"), pred_centroid_vox.astype(np.float32))
        np.save(os.path.join(out_case_dir, "crop_center_vox.npy"), center_vox.astype(np.float32))

        case_meta: Dict = {
            "case_id": case_id,
            "crop_size": list(crop_size),
            "wb_shape": [int(i) for i in wb.shape[:3]],
            "gt_centroid_vox": gt_centroid_vox.astype(float).tolist(),
            "pred_centroid_vox": pred_centroid_vox.astype(float).tolist(),
            "mask_centroid_vox": mask_centroid_vox.astype(float).tolist(),
            "crop_center_vox": center_vox.astype(float).tolist(),
            "crop_start_dhw": crop_start.astype(int).tolist(),
            "crop_end_dhw": crop_end.astype(int).tolist(),
            "loc_err_vox": loc_err,
            "image_path": os.path.join(out_case_dir, "image.npy"),
            "label_path": os.path.join(out_case_dir, "label.npy"),
        }
        with open(os.path.join(out_case_dir, "meta.json"), "w") as f:
            json.dump(case_meta, f, indent=2)
        stats.append(loc_err)

    summary = {
        "num_cases": len(case_ids),
        "crop_size": list(crop_size),
        "mean_loc_err_vox": float(np.mean(stats)) if stats else None,
        "median_loc_err_vox": float(np.median(stats)) if stats else None,
        "max_loc_err_vox": float(np.max(stats)) if stats else None,
        "min_loc_err_vox": float(np.min(stats)) if stats else None,
        "source_checkpoint": args.checkpoint,
        "prepared_dir": args.prepared_dir,
        "center_mode": args.center_mode,
        "mix_alpha": float(args.mix_alpha),
        "max_shift_vox": float(args.max_shift_vox),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "case_ids.json"), "w") as f:
        json.dump(case_ids, f, indent=2)

    print("[Done] Precompute finished")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
