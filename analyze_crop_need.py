"""
分析: 定位误差 vs 所需 crop_size.

核心逻辑:
  - 模型在 128³ 下采样空间上做 differentiable_crop_3d
  - GT annotation 是 48³ (在 128-空间中占 48 voxels)
  - crop 以 predicted centroid 为中心
  - 若定位误差为 err_128 (128-空间 voxels), 则:
      min_crop_size = 48 + 2 * max(err_128_per_axis)
    才能完全包住 GT 48³ 区域
"""

import argparse
import numpy as np
import yaml
import torch

from data.dataset import TNLocSegDataset, get_train_val_split
from models.locseg_net import LocSegNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # 加载模型
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
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    print(f"模型加载: {args.checkpoint} (phase={ckpt.get('phase')}, epoch={ckpt.get('epoch')})")

    # 验证集
    _, val_ids = get_train_val_split(
        cfg["data"]["prepared_dir"],
        train_ratio=cfg["data"]["train_val_split"],
        seed=cfg["data"]["random_seed"],
    )
    val_dataset = TNLocSegDataset(
        cfg["data"]["prepared_dir"], case_ids=val_ids, phase=2, lr_flip_prob=0.0,
    )
    print(f"验证集: {len(val_dataset)} cases\n")

    # 收集每个 case 的误差
    input_size = cfg["localization"]["input_size"]  # [128, 128, 128]
    D128 = input_size[0]  # 128

    all_err_128 = []       # per-axis error in 128-space (N, 3)
    all_max_err_128 = []   # max per-axis error per case
    all_min_crop = []      # minimum crop_size per case

    print(f"{'case_id':<25} {'err_d0':>7} {'err_d1':>7} {'err_d2':>7} {'max':>7} {'min_crop':>10}")
    print("-" * 70)

    with torch.no_grad():
        for idx in range(len(val_dataset)):
            sample = val_dataset[idx]
            case_id = sample["case_id"]
            wb_tensor = sample["whole_brain"]
            gt_norm = sample["centroid_norm"].numpy()

            # 推理定位
            wb = wb_tensor.unsqueeze(0).to(device)
            gt_t = torch.from_numpy(gt_norm).unsqueeze(0).float().to(device)
            loc_result = model.forward_loc(wb, gt_centroid_norm=gt_t)
            pred_norm = loc_result["centroid_norm"].cpu().squeeze(0).numpy()

            # 误差在归一化空间
            err_norm = np.abs(gt_norm - pred_norm)  # (3,)

            # 转换到 128-空间 (voxels)
            # differentiable_crop_3d 在 D=128 的体积上工作
            # 误差 = err_norm * (D - 1), 因为 align_corners=True
            err_128 = err_norm * (D128 - 1)  # (3,)

            max_err = err_128.max()
            # 最小 crop_size = 48 + 2 * max_per_axis_error
            min_crop = 48 + 2 * max_err

            all_err_128.append(err_128)
            all_max_err_128.append(max_err)
            all_min_crop.append(min_crop)

            print(f"{case_id:<25} {err_128[0]:7.2f} {err_128[1]:7.2f} {err_128[2]:7.2f} "
                  f"{max_err:7.2f} {min_crop:10.1f}")

    all_err_128 = np.array(all_err_128)  # (N, 3)
    all_max_err_128 = np.array(all_max_err_128)
    all_min_crop = np.array(all_min_crop)

    print(f"\n{'='*70}")
    print(f"  定位误差统计 (128-空间 voxels)")
    print(f"{'='*70}")
    dim_names = ["Dim0(L-R)", "Dim1(P-A)", "Dim2(I-S)"]
    for i, dn in enumerate(dim_names):
        vals = all_err_128[:, i]
        print(f"  {dn}: mean={vals.mean():.2f}, median={np.median(vals):.2f}, "
              f"max={vals.max():.2f}, P95={np.percentile(vals, 95):.2f}")

    print(f"\n  Max per-axis error (per case):")
    print(f"    mean={all_max_err_128.mean():.2f}, median={np.median(all_max_err_128):.2f}, "
          f"max={all_max_err_128.max():.2f}")
    print(f"    P90={np.percentile(all_max_err_128, 90):.2f}, "
          f"P95={np.percentile(all_max_err_128, 95):.2f}, "
          f"P99={np.percentile(all_max_err_128, 99):.2f}")

    print(f"\n{'='*70}")
    print(f"  所需最小 crop_size = 48 + 2 * max_axis_error")
    print(f"{'='*70}")
    print(f"  覆盖 50% cases (median): crop >= {np.percentile(all_min_crop, 50):.0f}")
    print(f"  覆盖 75% cases (P75):    crop >= {np.percentile(all_min_crop, 75):.0f}")
    print(f"  覆盖 90% cases (P90):    crop >= {np.percentile(all_min_crop, 90):.0f}")
    print(f"  覆盖 95% cases (P95):    crop >= {np.percentile(all_min_crop, 95):.0f}")
    print(f"  覆盖 100% cases (max):   crop >= {all_min_crop.max():.0f}")

    print(f"\n  常见 crop_size 覆盖率:")
    for cs in [48, 56, 64, 72, 80, 96]:
        pct = np.mean(all_min_crop <= cs) * 100
        print(f"    crop_size={cs}: 覆盖 {pct:.1f}% cases")

    print(f"\n  推荐: 使用 crop_size=64 可覆盖 {np.mean(all_min_crop <= 64)*100:.0f}% cases,")
    print(f"        使用 crop_size=80 可覆盖 {np.mean(all_min_crop <= 80)*100:.0f}% cases")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
