"""
在 nnU-Net 基础上运行自定义 loc+seg trainer 的入口脚本.

示例:
  export nnUNet_raw=${TN_ROOT}/nnUNet_data/nnUNet_raw
  export nnUNet_preprocessed=${TN_ROOT}/nnUNet_data/nnUNet_preprocessed
  export nnUNet_results=${TN_ROOT}/nnUNet_data/nnUNet_results

  conda run -n tnproject python train_nnunet_loc.py \
    --dataset Dataset001_TN \
    --config 3d_fullres \
    --fold 0 \
    --gpu 0
"""
import os
REPO_ROOT = os.environ.get('TN_ROOT', os.path.dirname(os.path.abspath(__file__)))

import argparse
import os
import torch

from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

from nnunet_loc_trainer import nnUNetTrainerLoc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Dataset001_TN")
    parser.add_argument("--config", type=str, default="3d_fullres")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--plans", type=str, default="nnUNetPlans")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument("--pretrained_weights", type=str, default="",
                        help="warm-start: 加载已有 checkpoint 的网络权重作为初始化, "
                             "但重置 epoch/LR scheduler (strict=False, 允许新增参数如 side_head)")
    parser.add_argument("--reset_seg_head", action="store_true",
                        help="在加载 pretrained_weights 后重置分割输出头，避免继承全背景塌缩")
    parser.add_argument("--reset_seg_head_bg_bias", type=float, default=0.0,
                        help="reset_seg_head 时背景通道初始 bias，默认 0.0")
    parser.add_argument("--loc_arch", choices=["attn", "se"], default="attn",
                        help="Loc encoder 变体: 'attn' (默认, AttentionGate3D spatial attention) "
                             "或 'se' (Squeeze-and-Excitation channel attention, based on 2025 J Neurosurg SOTA)")
    parser.add_argument("--seg_se", action="store_true",
                        help="为分割 UNet 的 encoder/decoder 每个 stage 注入 SE 通道注意力")
    parser.add_argument("--seg_se_reduction", type=int, default=16,
                        help="分割 UNet SE block 的 reduction ratio (默认 16)")
    parser.add_argument("--smooth_sigma", type=float, default=0.0,
                        help="Gaussian smoothing sigma (voxel) 应用于 whole-brain 输入, 0=关闭")
    parser.add_argument("--affine_prob", type=float, default=None,
                        help="Phase1 affine augmentation 概率. 默认沿用 trainer 的 0.4. "
                             "对 centroid 定位任务建议设 0.0")
    parser.add_argument("--prepared_data_dir", type=str, default=os.path.join(REPO_ROOT, 'prepared_data'))
    parser.add_argument("--loc_input_size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--phase1_epochs", type=int, default=150)
    parser.add_argument("--phase2_epochs", type=int, default=250)
    parser.add_argument("--phase3_epochs", type=int, default=600)
    parser.add_argument("--phase3_warmup_epochs", type=int, default=40)
    parser.add_argument(
        "--phase3_fg_aux_weight",
        type=float,
        default=None,
        help="phase3 前景 vs 背景 BCE 辅助项系数; 不设则走 TN_PHASE3_FG_AUX_WEIGHT 或 trainer 默认",
    )
    parser.add_argument(
        "--phase3_min_gt_mix",
        type=float,
        default=None,
        help="phase3 质心混合中最小 GT 权重; 不设则走 TN_PHASE3_MIN_GT_MIX 或 0.2",
    )
    parser.add_argument(
        "--phase3_seg_lr",
        type=float,
        default=None,
        help="phase3 分割 UNet 的 poly 初值 (与 loc 可分开设); 不设则沿用 plans 的 initial_lr (常见 0.01)",
    )
    parser.add_argument("--loc_loss_weight", type=float, default=1.0)
    parser.add_argument("--loc_sigma", type=float, default=3.5,
                        help="Gaussian heatmap sigma (以 96³ 为参考分辨率). "
                             "effective_sigma = loc_sigma * (loc_input_size / 96)")
    parser.add_argument("--loc_lr", type=float, default=1e-3)
    parser.add_argument("--loc_phase1_lr", type=float, default=None)
    parser.add_argument("--loc_phase3_lr", type=float, default=5e-4)
    parser.add_argument("--loc_phase3_lr_schedule", type=str, default="constant",
                        choices=["constant", "cosine", "poly"],
                        help="Phase3 loc LR schedule: constant/cosine/poly")
    parser.add_argument("--loc_phase3_lr_min_ratio", type=float, default=0.1,
                        help="Phase3 cosine 最低 LR 比例, lr_min = loc_phase3_lr * ratio")
    parser.add_argument("--loc_phase1_weight_decay", type=float, default=1e-5)
    parser.add_argument("--loc_phase1_lr_schedule", type=str, default="cosine",
                        choices=["cosine", "poly"],
                        help="Phase1 loc LR schedule: 'cosine' (默认) 或 'poly'")
    parser.add_argument("--loc_phase1_lr_min_ratio", type=float, default=0.01,
                        help="Cosine schedule 最低 LR 比例, lr_min = lr * ratio (默认 0.01)")
    parser.add_argument("--case_cache_size", type=int, default=24)
    parser.add_argument("--loc_bs_phase1", type=int, default=8)
    parser.add_argument(
        "--phase1_case_filter",
        type=str,
        default="",
        choices=["", "openneuro", "inhouse"],
        help="Phase1 train/val 子集: openneuro=仅 sub-*; inhouse=排除 sub-*; 空=全量",
    )
    parser.add_argument("--seg_bs", type=int, default=0,
                        help="phase2/3 的 nnU-Net seg batch size 覆盖值, <=0 表示沿用 plans")
    parser.add_argument("--seg_patch_size", type=int, nargs=3, default=None,
                        help="phase2/3 的 nnU-Net seg patch 尺寸覆盖值, 例如: --seg_patch_size 48 48 48")
    parser.add_argument("--inner_dice_crop", type=int, default=48,
                        help="在 seg patch 内按 GT 前景质心计算 inner dice 的裁剪尺寸 (默认 48)")
    parser.add_argument("--disable_inner_dice", action="store_true",
                        help="关闭 inner dice 统计")
    parser.add_argument(
        "--wb_target_spacing",
        type=str,
        default="",
        help="全脑重采样目标 spacing (mm): '1' 或 '1,1,1'。空则禁用。亦可 export TN_WB_TARGET_SPACING",
    )
    parser.add_argument(
        "--wb_use_plans_fg_norm",
        action="store_true",
        help="用 nnU-Net plans 指纹中的 foreground mean/std 归一化全脑。亦可 export TN_WB_USE_PLANS_FG_NORM=1",
    )
    parser.add_argument(
        "--oversample_foreground_percent",
        type=float,
        default=0.85,
        help="nnU-Net patch 前景过采样比例 [0,1]. 默认 0.85 (针对极小 FG 调高, "
             "Sudre/Hashemi 推荐 ≥0.8). 若已 export TN_OVERSAMPLE_FOREGROUND_PERCENT 则后者优先.",
    )
    parser.add_argument(
        "--loss",
        choices=["default", "focal_tversky"],
        default="default",
        help="主分割 loss. default = DC+CE (nnU-Net 原版); "
             "focal_tversky = CE + FocalTversky, 针对极小 FG 偏向召回, "
             "对应 env TN_LOSS.",
    )
    parser.add_argument(
        "--ft_alpha",
        type=float,
        default=0.7,
        help="FocalTversky alpha (FN 惩罚, tiny FG 默认 0.7). 仅在 --loss focal_tversky 时生效.",
    )
    parser.add_argument(
        "--ft_beta",
        type=float,
        default=0.3,
        help="FocalTversky beta (FP 惩罚, tiny FG 默认 0.3). 仅在 --loss focal_tversky 时生效.",
    )
    parser.add_argument(
        "--ft_gamma",
        type=float,
        default=4.0 / 3.0,
        help="FocalTversky gamma (难样本聚焦, 默认 4/3 ≈ 1.33). 仅在 --loss focal_tversky 时生效.",
    )
    parser.add_argument(
        "--seg_dropout_p",
        type=float,
        default=None,
        help="分割 PlainConvUNet 使用 Dropout3d 的 p，如 0.15。不设则无 dropout。亦可 export TN_SEG_DROPOUT_P",
    )
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--num_iters_per_epoch", type=int, default=100)
    parser.add_argument("--num_val_iters_per_epoch", type=int, default=50)
    parser.add_argument("--num_iters_phase3", type=int, default=None,
                        help="phase3 单独 train iter 数, 不设则沿用 num_iters_per_epoch")
    parser.add_argument("--num_val_iters_phase3", type=int, default=None,
                        help="phase3 单独 val iter 数, 不设则沿用 num_val_iters_per_epoch")
    parser.add_argument("--phase3_val_full", action="store_true",
                        help="phase3 val 启用全覆盖模式: iters 自动调到 ceil(n_val_cases*cover/bs), "
                             "覆盖整个 val 集 (默认每 case 平均 4 patch). "
                             "替代 --num_val_iters_phase3 的固定小采样, pseudo dice 标准差从 ±0.03 降到 ±0.01.")
    parser.add_argument("--phase3_val_cover", type=int, default=4,
                        help="--phase3_val_full 模式下每个 val case 平均采样的 patch 数 (默认 4).")
    parser.add_argument("--quick_test", action="store_true",
                        help="启用快速测试配置: loc_input=64, loc_bs=4, train_iters=40, val_iters=3, val_interval=5")
    args = parser.parse_args()

    # 为当前项目默认关闭 torch.compile, 避免动态 shape + whole-brain 路径导致首轮极慢/重编译
    os.environ.setdefault("nnUNet_compile", "f")

    if args.quick_test:
        args.loc_input_size = [64, 64, 64]
        args.loc_bs_phase1 = 4
        args.num_iters_per_epoch = 40
        args.num_val_iters_per_epoch = 3
        args.val_interval = 5
        print("[QuickTest] loc_input=64^3, loc_bs_phase1=4, train_iters=40, val_iters=3, val_interval=5")

    assert nnUNet_preprocessed is not None, "nnUNet_preprocessed 未设置"

    dataset_name = maybe_convert_to_dataset_name(args.dataset)
    preprocessed_base = join(nnUNet_preprocessed, dataset_name)
    plans_file = join(preprocessed_base, args.plans + ".json")
    dataset_json_file = join(preprocessed_base, "dataset.json")
    assert isfile(plans_file), f"plans 文件不存在: {plans_file}"
    assert isfile(dataset_json_file), f"dataset.json 不存在: {dataset_json_file}"

    plans = load_json(plans_file)
    dataset_json = load_json(dataset_json_file)
    if args.seg_patch_size is not None:
        if args.config not in plans.get("configurations", {}):
            raise KeyError(f"配置 {args.config} 不在 plans 中")
        if any(i <= 0 for i in args.seg_patch_size):
            raise ValueError(f"seg_patch_size 必须为正整数, got={args.seg_patch_size}")
        old_patch = plans["configurations"][args.config]["patch_size"]
        plans["configurations"][args.config]["patch_size"] = [int(i) for i in args.seg_patch_size]
        print(f"[seg-patch] override patch_size {old_patch} -> {plans['configurations'][args.config]['patch_size']}")

    os.environ["TN_PREPARED_DATA_DIR"] = args.prepared_data_dir
    os.environ["TN_LOC_ARCH"] = args.loc_arch
    os.environ["TN_SEG_SE"] = "1" if args.seg_se else "0"
    os.environ["TN_SEG_SE_REDUCTION"] = str(args.seg_se_reduction)
    os.environ["TN_SMOOTH_SIGMA"] = str(args.smooth_sigma)
    if args.affine_prob is not None:
        os.environ["TN_PHASE1_AFFINE_PROB"] = str(args.affine_prob)
    os.environ["TN_LOC_INPUT_SIZE"] = ",".join(str(i) for i in args.loc_input_size)
    os.environ["TN_PHASE1_EPOCHS"] = str(args.phase1_epochs)
    os.environ["TN_PHASE2_EPOCHS"] = str(args.phase2_epochs)
    os.environ["TN_PHASE3_EPOCHS"] = str(args.phase3_epochs)
    os.environ["TN_PHASE3_WARMUP_EPOCHS"] = str(args.phase3_warmup_epochs)
    if args.phase3_fg_aux_weight is not None:
        os.environ["TN_PHASE3_FG_AUX_WEIGHT"] = str(args.phase3_fg_aux_weight)
    if args.phase3_min_gt_mix is not None:
        os.environ["TN_PHASE3_MIN_GT_MIX"] = str(args.phase3_min_gt_mix)
    if args.phase3_seg_lr is not None:
        os.environ["TN_PHASE3_SEG_LR"] = str(args.phase3_seg_lr)
    os.environ["TN_LOC_LOSS_WEIGHT"] = str(args.loc_loss_weight)
    os.environ["TN_LOC_SIGMA"] = str(args.loc_sigma)
    os.environ["TN_LOC_LR"] = str(args.loc_lr)
    os.environ["TN_LOC_PHASE1_LR"] = str(args.loc_phase1_lr if args.loc_phase1_lr is not None else args.loc_lr)
    os.environ["TN_LOC_PHASE3_LR"] = str(args.loc_phase3_lr)
    os.environ["TN_LOC_PHASE3_LR_SCHEDULE"] = args.loc_phase3_lr_schedule
    os.environ["TN_LOC_PHASE3_LR_MIN_RATIO"] = str(args.loc_phase3_lr_min_ratio)
    os.environ["TN_LOC_PHASE1_WEIGHT_DECAY"] = str(args.loc_phase1_weight_decay)
    os.environ["TN_LOC_PHASE1_LR_SCHEDULE"] = args.loc_phase1_lr_schedule
    os.environ["TN_LOC_PHASE1_LR_MIN_RATIO"] = str(args.loc_phase1_lr_min_ratio)
    os.environ["TN_CASE_CACHE_SIZE"] = str(args.case_cache_size)
    os.environ["TN_LOC_BS_PHASE1"] = str(args.loc_bs_phase1)
    if getattr(args, "phase1_case_filter", ""):
        os.environ["TN_PHASE1_CASE_FILTER"] = args.phase1_case_filter
    else:
        os.environ.pop("TN_PHASE1_CASE_FILTER", None)
    os.environ["TN_SEG_BS"] = str(args.seg_bs)
    os.environ["TN_INNER_DICE_ENABLE"] = "0" if args.disable_inner_dice else "1"
    os.environ["TN_INNER_DICE_CROP"] = str(args.inner_dice_crop)
    os.environ["TN_VAL_INTERVAL"] = str(args.val_interval)
    os.environ["TN_NUM_ITERS_PER_EPOCH"] = str(args.num_iters_per_epoch)
    os.environ["TN_NUM_VAL_ITERS_PER_EPOCH"] = str(args.num_val_iters_per_epoch)
    if args.num_iters_phase3 is not None:
        os.environ["TN_NUM_ITERS_PER_EPOCH_PHASE3"] = str(args.num_iters_phase3)
    if args.num_val_iters_phase3 is not None:
        os.environ["TN_NUM_VAL_ITERS_PER_EPOCH_PHASE3"] = str(args.num_val_iters_phase3)
    if args.phase3_val_full:
        os.environ["TN_PHASE3_VAL_FULL"] = "1"
        os.environ["TN_PHASE3_VAL_COVER"] = str(args.phase3_val_cover)

    if getattr(args, "wb_target_spacing", "").strip():
        os.environ["TN_WB_TARGET_SPACING"] = args.wb_target_spacing.strip()
    if getattr(args, "wb_use_plans_fg_norm", False):
        os.environ["TN_WB_USE_PLANS_FG_NORM"] = "1"
    if getattr(args, "oversample_foreground_percent", None) is not None:
        # 不覆盖已 export 的环境变量, 保护脚本级显式设定
        os.environ.setdefault(
            "TN_OVERSAMPLE_FOREGROUND_PERCENT", str(args.oversample_foreground_percent)
        )
    if getattr(args, "loss", "default") and args.loss != "default":
        os.environ["TN_LOSS"] = args.loss
        os.environ["TN_FT_ALPHA"] = str(args.ft_alpha)
        os.environ["TN_FT_BETA"] = str(args.ft_beta)
        os.environ["TN_FT_GAMMA"] = str(args.ft_gamma)
    if getattr(args, "seg_dropout_p", None) is not None:
        os.environ["TN_SEG_DROPOUT_P"] = str(args.seg_dropout_p)

    device = torch.device(type="cuda", index=args.gpu) if torch.cuda.is_available() else torch.device("cpu")
    trainer = nnUNetTrainerLoc(
        plans=plans,
        configuration=args.config,
        fold=args.fold,
        dataset_json=dataset_json,
        device=device,
    )

    if args.continue_training:
        ckpt_final = join(trainer.output_folder, "checkpoint_final.pth")
        ckpt_latest = join(trainer.output_folder, "checkpoint_latest.pth")
        ckpt_best = join(trainer.output_folder, "checkpoint_best.pth")
        # 优先 latest，避免存在 checkpoint_final 时直接判定训练已完成
        ckpt = ckpt_latest if isfile(ckpt_latest) else (ckpt_final if isfile(ckpt_final) else (ckpt_best if isfile(ckpt_best) else None))
        if ckpt is not None:
            print(f"[Resume] {ckpt}")
            try:
                trainer.load_checkpoint(ckpt)
            except (ValueError, KeyError, AssertionError) as e:
                # 常见不兼容场景:
                # 1) optimizer 参数组数量不同
                # 2) 自定义/精简 checkpoint 缺失 logging 字段
                # 3) logger 历史长度与 current_epoch 不一致
                msg = str(e)
                can_fallback = (
                    ("different number of parameter groups" in msg)
                    or ("'logging'" in msg)
                    or ("logging lists length is off" in msg)
                )
                if not can_fallback:
                    raise
                print(f"[Resume-Fallback] {type(e).__name__}: {msg}")
                print("[Resume-Fallback] load network weights only")
                ckpt_obj = torch.load(ckpt, map_location=device, weights_only=False)
                trainer.network.load_state_dict(ckpt_obj["network_weights"])
                trainer.current_epoch = int(ckpt_obj.get("current_epoch", 0))
                print(f"[Resume-Fallback] restored epoch={trainer.current_epoch}")
        else:
            print("[Resume] 未找到可续训 checkpoint, 将从头训练")
    elif args.pretrained_weights:
        # Warm-start: 用已有 checkpoint 的权重初始化, 但 epoch/LR/optimizer 都重新开始
        # strict=False 允许新增的参数 (如 side_head) 随机初始化
        if not isfile(args.pretrained_weights):
            raise FileNotFoundError(f"pretrained_weights not found: {args.pretrained_weights}")
        print(f"[WarmStart] loading weights from {args.pretrained_weights}")
        ckpt_obj = torch.load(args.pretrained_weights, map_location=device, weights_only=False)
        state_dict = ckpt_obj["network_weights"]
        # 先初始化 trainer 内部 network (否则 load_state_dict 找不到模块)
        trainer.initialize()
        missing, unexpected = trainer.network.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WarmStart] missing keys (will be randomly initialized): {len(missing)}")
            for k in missing:
                print(f"  - {k}")
        if unexpected:
            print(f"[WarmStart] unexpected keys (ignored): {len(unexpected)}")
            for k in unexpected:
                print(f"  - {k}")
        if args.reset_seg_head:
            trainer.reset_segmentation_heads(bg_bias=args.reset_seg_head_bg_bias)
            print(f"[WarmStart] reset segmentation head (bg_bias={args.reset_seg_head_bg_bias})")
        print("[WarmStart] epoch/LR/optimizer reset; training from epoch 0")

    trainer.run_training()


if __name__ == "__main__":
    main()
