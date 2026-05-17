"""
eval_phase3_val_dice.py
=======================
对 phase3_best.pth 在 val split 上跑一次大规模 validation
(默认 200 iters vs 训练时 30 iters), 算出低噪音的 pseudo dice 估计。

复用 nnUNetTrainerLoc 的全部机制 (loc forward / GT-crop / seg forward /
hard dice 累积), 因此结果与训练日志里的 "Pseudo dice" 是同一指标,
只是采样次数从 30 涨到 200, 标准差从 ±0.03 降到 ±0.012 左右。

用法 (在 mix0604 GPU 之外的卡上跑):
  CUDA_VISIBLE_DEVICES=0 conda run -n tnproject python eval_phase3_val_dice.py \
      --ckpt ${TN_ROOT}/nnUNet_data/nnUNet_results_p3_scratch_mix0604_20260502/Dataset001_TN/nnUNetTrainerLoc__nnUNetPlans__3d_fullres/fold_0/phase3_best.pth \
      --num_val_iters 200
"""
import os
REPO_ROOT = os.environ.get('TN_ROOT', os.path.dirname(os.path.abspath(__file__)))

import argparse
import os
import sys
import time

ROOT = "${TN_ROOT}"
sys.path.insert(0, ROOT)

# ---------------- env: 必须和 mix0604 训练时一致 (架构相关) ----------------
# 这些环境变量在 trainer.__init__ 里被读取, 必须在 import nnunet_loc_trainer 之前设置
os.environ.setdefault("TN_SEG_DROPOUT_P", "0.15")          # 决定 conv block 结构
os.environ.setdefault("TN_LOC_ARCH", "attn")
os.environ.setdefault("TN_PHASE3_VAL_USE_GT_CROP", "1")    # GT-centered crop, 与训练 val 一致
os.environ.setdefault("TN_WB_TARGET_SPACING", "1,1,1")
os.environ.setdefault("TN_WB_USE_PLANS_FG_NORM", "1")
os.environ.setdefault("TN_OVERSAMPLE_FOREGROUND_PERCENT", "0.55")
os.environ.setdefault("TN_VESSEL_LOSS_WEIGHT", "0.75")
os.environ.setdefault("TN_PHASE3_VESSEL_OVERSAMPLE_PROB", "0.35")

# 训练流程相关 (影响 phase 判定)
os.environ.setdefault("TN_PHASE1_EPOCHS", "0")
os.environ.setdefault("TN_PHASE2_EPOCHS", "0")
os.environ.setdefault("TN_PHASE3_EPOCHS", "1000")
os.environ.setdefault("TN_PHASE3_WARMUP_EPOCHS", "200")
os.environ.setdefault("TN_PHASE3_MIN_GT_MIX", "0.6")       # mix0604 (val 用 GT crop, 此项不影响 val 结果)
os.environ.setdefault("TN_LOC_LOSS_WEIGHT", "0.2")
os.environ.setdefault("TN_LOC_PHASE3_LR", "5e-4")
os.environ.setdefault("TN_LOC_PHASE3_LR_SCHEDULE", "cosine")
os.environ.setdefault("TN_LOC_PHASE3_LR_MIN_RATIO", "0.1")

# nnUNet 路径
# 重要: nnUNet_results 用一个独立的 eval 目录, 避免 trainer 在 mix0604 的目录下
# 写训练日志 / 覆盖 checkpoint。dataset/plans/splits 都在 nnUNet_preprocessed 里, 不受影响。
os.environ.setdefault("nnUNet_results", f"{ROOT}/nnUNet_data/nnUNet_results_eval_tmp")
os.environ.setdefault("nnUNet_raw", f"{ROOT}/nnUNet_data/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed", f"{ROOT}/nnUNet_data/nnUNet_preprocessed")

# ---------------- imports (env 已就绪) ----------------
import torch
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile

from nnunet_loc_trainer import nnUNetTrainerLoc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="phase3_best.pth path")
    parser.add_argument("--dataset", default="Dataset001_TN")
    parser.add_argument("--config", default="3d_fullres")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--num_val_iters", type=int, default=200,
                        help="validation iter 数 (训练时为 30, 200 让方差降到 ±0.012)")
    parser.add_argument("--seg_patch_size", type=int, nargs=3, default=[96, 96, 96])
    args = parser.parse_args()

    if not isfile(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    # 让 trainer 在构造时就读取这个值
    os.environ["TN_NUM_VAL_ITERS_PER_EPOCH_PHASE3"] = str(args.num_val_iters)

    device = torch.device("cuda:0")  # 单卡, GPU 由外层 CUDA_VISIBLE_DEVICES 控制
    print(f"[eval] device={device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '?')}")

    # 加载 plans + dataset_json (与 train_nnunet_loc.py 同样路径)
    preprocessed_dir = join(os.environ["nnUNet_preprocessed"], args.dataset)
    plans_file = join(preprocessed_dir, f"{args.plans}.json")
    dataset_json = load_json(join(preprocessed_dir, "dataset.json"))
    plans = load_json(plans_file)

    # 覆盖 seg patch size (mix0604 启动时也是这么改的, 见 train_nnunet_loc.py:175)
    old_ps = plans["configurations"][args.config]["patch_size"]
    plans["configurations"][args.config]["patch_size"] = [int(i) for i in args.seg_patch_size]
    print(f"[eval] seg patch_size override: {old_ps} -> {args.seg_patch_size}")

    # 构造 trainer
    trainer = nnUNetTrainerLoc(
        plans, args.config, args.fold, dataset_json, device=device
    )
    trainer.initialize()

    # 加载 ckpt (架构应完全匹配, strict=False 容忍 deep supervision wrapper 之类的轻微差异)
    print(f"[eval] loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["network_weights"]
    missing, unexpected = trainer.network.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[eval][WARN] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        for k in missing[:5]: print(f"    missing: {k}")
        for k in unexpected[:5]: print(f"    unexpected: {k}")
    else:
        print("[eval] ckpt loaded with 0 missing / 0 unexpected keys (架构完全匹配)")
    print(f"[eval] ckpt epoch={ckpt.get('current_epoch', '?')}")

    # 把 trainer 推进到 phase3
    target_epoch = trainer.phase1_epochs + trainer.phase2_epochs  # = 0 + 0 = 0 for our setup
    trainer.current_epoch = max(target_epoch, int(ckpt.get("current_epoch", target_epoch)))
    print(f"[eval] phase = {trainer._get_phase_name()} @ epoch={trainer.current_epoch}")
    print(f"[eval] num_val_iters (phase3) = {trainer.num_val_iterations_per_epoch_phase3}")

    # 创建 dataloader_train / dataloader_val
    # nnUNet 默认在 on_train_start 里调 get_dataloaders, 这里手动调以跳过 on_train_start 的其它副作用
    print("[eval] building dataloaders...")
    trainer.dataloader_train, trainer.dataloader_val = trainer.get_dataloaders()
    print("[eval] dataloaders ready")

    # 跑一次 validation
    print(f"\n[eval] running {args.num_val_iters} val iters...")
    t0 = time.time()
    trainer.network.eval()
    with torch.no_grad():
        trainer.on_validation_epoch_start()
        val_outputs = []
        for i in range(args.num_val_iters):
            batch = next(trainer.dataloader_val)
            val_outputs.append(trainer.validation_step(batch))
            if (i + 1) % 25 == 0:
                print(f"    [{i+1}/{args.num_val_iters}] iters done, "
                      f"elapsed={time.time()-t0:.0f}s")
        trainer.on_validation_epoch_end(val_outputs)
    elapsed = time.time() - t0
    print(f"[eval] done in {elapsed:.0f}s")

    # 读取 nnUNet logger 的 mean_fg_dice / dice_per_class
    print("\n=========== EVAL RESULTS ===========")
    log = trainer.logger.my_fantastic_logging
    if log["val_losses"]:
        print(f"  val_loss            : {log['val_losses'][-1]:.4f}")
    if log["mean_fg_dice"]:
        print(f"  mean_fg_dice (硬)   : {log['mean_fg_dice'][-1]:.4f}")
    if log["dice_per_class_or_region"]:
        dpc = log["dice_per_class_or_region"][-1]
        print(f"  dice_per_class      : {dpc}")

    print(f"\n  num_val_iters       : {args.num_val_iters}")
    print(f"  ckpt                : {args.ckpt}")
    print(f"  ckpt epoch          : {ckpt.get('current_epoch', '?')}")
    print("====================================\n")


if __name__ == "__main__":
    main()