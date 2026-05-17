"""
train.py
========
LocSegNet training entry point.

Usage:
    # === Single-GPU ===
    # Full three-phase training (GPU from config or auto)
    python train.py --config configs/default.yaml

    # Override GPU (omit or auto = pick GPU with most free VRAM)
    python train.py --config configs/default.yaml --gpu_id 1
    python train.py --config configs/default.yaml --gpu_id auto

    # Custom run name
    python train.py --config configs/default.yaml --run_name exp_lr1e4

    # Start from Phase 2 (load Phase 1 weights)
    python train.py --config configs/default.yaml --start_phase 2 --checkpoint checkpoints/phase1_final.pth

    # Run Phase 1 only
    python train.py --config configs/default.yaml --only_phase 1

    # Resume: restore optimizer/scheduler/epoch; reuses original run dir
    python train.py --config configs/default.yaml --resume runs/run_20260210_153000/checkpoints/phase1_epoch60.pth

    # === Multi-GPU DDP ===
    CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 train.py --config configs/default.yaml

    torchrun --nproc_per_node=2 train.py --config configs/default.yaml

    CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 train.py --config configs/default.yaml --resume runs/run_xxx/checkpoints/phase2_epoch100.pth

    # Single-GPU again: use plain python; checkpoints are compatible
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime

import torch
import torch.distributed as dist
import yaml

# Project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.locseg_net import LocSegNet
from data.dataset import TNLocSegDataset, get_train_val_split, locseg_collate_fn
from training.trainer import LocSegTrainer


def pick_idle_gpu():
    """Physical GPU index with largest nvidia-smi memory.free (MiB)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    best_free = None
    best_idx = 0
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
            free_mib = int(parts[1])
        except ValueError:
            continue
        if best_free is None or free_mib > best_free:
            best_free = free_mib
            best_idx = idx
    return best_idx


def resolve_gpu_id(raw):
    """Resolve device.gpu_id: None/auto -> pick idle GPU; int or numeric str -> fixed GPU."""
    if raw is None:
        return pick_idle_gpu()
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s == "auto" or s == "":
            return pick_idle_gpu()
        return int(s, 10)
    return int(raw)


# ================================================================
# DDP helpers (torchrun sets env vars)
# ================================================================
def setup_distributed():
    """
    Detect and init distributed training.

    torchrun sets LOCAL_RANK / RANK / WORLD_SIZE.
    Plain python: env vars absent -> non-distributed.

    Returns:
        (distributed, local_rank, rank, world_size)
    """
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 0, 1  # not distributed

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    return True, local_rank, rank, world_size


def cleanup_distributed():
    """Tear down distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="LocSegNet training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config")
    parser.add_argument("--start_phase", type=int, default=1,
                        help="Start phase (1/2/3)")
    parser.add_argument("--only_phase", type=int, default=0,
                        help="Run only this phase (1/2/3); 0 = all")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load weights only (no optimizer/epoch state)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume full state from checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda / cpu (default from config)")
    parser.add_argument(
        "--gpu_id",
        type=str,
        default=None,
        help="GPU id; omit or auto = pick GPU with most free VRAM (overrides config device.gpu_id)",
    )
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name (default: timestamp run_YYYYMMDD_HHMMSS)")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Explicit run directory (usually set by resume)")
    args = parser.parse_args()

    # --- Load config ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # --- Distributed (torchrun / plain python) ---
    distributed, local_rank, rank, world_size = setup_distributed()
    is_main = (rank == 0)

    # --- Device ---
    gpu_chosen_auto = False
    if distributed:
        # DDP: each process uses LOCAL_RANK GPU
        device = f"cuda:{local_rank}"
        gpu_id = local_rank
    else:
        # Single-GPU
        device_cfg = cfg.get("device", {})
        device_type = args.device or device_cfg.get("type", "cuda")
        raw_gpu = args.gpu_id if args.gpu_id is not None else device_cfg.get("gpu_id", "auto")
        try:
            gpu_id = resolve_gpu_id(raw_gpu)
        except (ValueError, TypeError) as e:
            raise SystemExit(f"Invalid device.gpu_id / --gpu_id: {raw_gpu!r}") from e
        gpu_chosen_auto = raw_gpu is None or (
            isinstance(raw_gpu, str) and raw_gpu.strip().lower() in ("auto", "")
        )

        if device_type == "cuda" and torch.cuda.is_available():
            device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
        elif device_type == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA unavailable, using CPU")
            device = "cpu"
        else:
            device = "cpu"

    # --- Run directory ---
    runs_dir = cfg["output"].get("runs_dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))

    if args.run_dir:
        run_dir = args.run_dir
    elif args.resume:
        # Infer run dir from checkpoint path: .../runs/run_xxx/checkpoints/phase1_epoch60.pth
        ckpt_abs = os.path.abspath(args.resume)
        ckpt_parent = os.path.dirname(ckpt_abs)  # checkpoints/
        candidate = os.path.dirname(ckpt_parent)  # run_xxx/
        if os.path.isdir(os.path.join(candidate, "checkpoints")):
            run_dir = candidate
            print(f"Resume: using run dir {run_dir}")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(runs_dir, f"run_{ts}_resumed")
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = args.run_name or ts
        run_dir = os.path.join(runs_dir, f"run_{name}")

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cfg["output"]["checkpoint_dir"] = ckpt_dir
    cfg["output"]["log_dir"] = log_dir

    config_copy_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_copy_path):
        shutil.copy2(args.config, config_copy_path)

    latest_link = os.path.join(runs_dir, "latest")
    try:
        if os.path.islink(latest_link):
            os.remove(latest_link)
        os.symlink(run_dir, latest_link)
    except OSError:
        pass  # symlink optional

    if is_main:
        print("=" * 60)
        print("LocSegNet - trigeminal localization + segmentation")
        print("=" * 60)
        print(f"Config:  {args.config}")
        print(f"Run dir: {run_dir}")
        print(f"Device:  {device}")
        if distributed:
            print(f"DDP: world_size={world_size}")
        if device.startswith("cuda"):
            if gpu_chosen_auto:
                print("Auto-selected physical GPU with most free VRAM (memory.free)")
            print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            print(f"VRAM: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")

    # --- Datasets ---
    prepared_dir = cfg["data"]["prepared_dir"]
    train_ids, val_ids = get_train_val_split(
        prepared_dir,
        train_ratio=cfg["data"]["train_val_split"],
        seed=cfg["data"]["random_seed"],
    )

    lr_flip_prob = cfg.get("augmentation", {}).get("loc_augmentation", {}).get("lr_flip_prob", 0.5)

    mask_size = cfg["segmentation"].get("mask_size", None)
    train_dataset = TNLocSegDataset(
        prepared_dir, case_ids=train_ids, phase=1,
        cache_wb=False,  # disable cache to avoid RAM blowout
        lr_flip_prob=lr_flip_prob,
        mask_size=mask_size,
    )
    val_dataset = TNLocSegDataset(
        prepared_dir, case_ids=val_ids, phase=1,
        cache_wb=False,
        lr_flip_prob=0.0,  # no aug on val
        mask_size=mask_size,
    )

    if is_main:
        print(f"Train: {len(train_dataset)} cases (LR flip prob={lr_flip_prob})")
        print(f"Val:   {len(val_dataset)} cases")
        print("Datasets ready, building model...")

    # --- Model ---
    dual_seg = cfg["segmentation"].get("dual_seg", False)
    seg_dropout = cfg["segmentation"].get("dropout", 0.0)
    seg_arch = str(cfg["segmentation"].get("seg_arch", "nnunet")).lower()
    if seg_arch not in ("nnunet", "plainconvunet"):
        raise ValueError(
            f"segmentation.seg_arch={seg_arch} is not supported. "
            "Use nnunet/plainconvunet."
        )
    if dual_seg:
        raise ValueError(
            "segmentation.dual_seg=True is not supported in nnU-Net-only mode."
        )

    seg_cfg = cfg.get("segmentation", {})
    model = LocSegNet(
        loc_input_size=tuple(cfg["localization"]["input_size"]),
        seg_crop_size=tuple(cfg["segmentation"]["crop_size"]),
        num_classes=cfg["segmentation"]["num_classes"],
        loc_channels=tuple(cfg["localization"]["encoder_channels"]),
        loc_arch=cfg["localization"].get("loc_arch", "nnunet"),
        loc_features_per_stage=tuple(cfg["localization"].get("features_per_stage", [16, 32, 64, 128, 256])),
        loc_n_conv_per_stage=cfg["localization"].get("n_conv_per_stage", 2),
        loc_n_conv_per_stage_decoder=cfg["localization"].get("n_conv_per_stage_decoder", 2),
        seg_channels=tuple(cfg["segmentation"].get("encoder_channels", [32, 64, 128, 256, 512])),
        deep_supervision=cfg["segmentation"]["deep_supervision"],
        dual_seg=dual_seg,
        dropout=seg_dropout,
        seg_arch=seg_arch,
        seg_features_per_stage=tuple(cfg["segmentation"].get("features_per_stage", [32, 64, 128, 256, 320])),
        n_conv_per_stage=cfg["segmentation"].get("n_conv_per_stage", 2),
        n_conv_per_stage_decoder=cfg["segmentation"].get("n_conv_per_stage_decoder", 2),
        use_neurovasc_modules=bool(seg_cfg.get("use_neurovasc_modules", False)),
        neurovasc_mscf=bool(seg_cfg.get("neurovasc_mscf", True)),
        neurovasc_cda_last_n=int(seg_cfg.get("neurovasc_cda_last_n", 2)),
        neurovasc_cda_stochastic_depth_p=float(
            seg_cfg.get("neurovasc_cda_stochastic_depth_p", 0.1)
        ),
        loc_head_select=str(cfg.get("localization", {}).get("loc_head_select", "gt")),
    )

    if is_main:
        loc_arch = cfg["localization"].get("loc_arch", "nnunet")
        loc_params = sum(p.numel() for p in model.loc_encoder.parameters())
        seg_params = sum(p.numel() for p in model.seg_network.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel parameters:")
        print(f"  Localization: {loc_params / 1e6:.2f} M  (arch={loc_arch})")
        print(f"  Segmentation: {seg_params / 1e6:.2f} M  (arch={seg_arch})")
        print(f"  Total:        {total_params / 1e6:.2f} M")
        if seg_arch == "nnunet":
            fps = cfg["segmentation"].get("features_per_stage", [32, 64, 128, 256, 320])
            print(f"  PlainConvUNet features: {fps}")
        if seg_cfg.get("use_neurovasc_modules", False):
            print(
                "  NeuroVasc paper blocks (Sec. 2.3.1–2.3.2): MSC^2F bottleneck + CDA^2F on last "
                f"{seg_cfg.get('neurovasc_cda_last_n', 2)} decoder stage(s); "
                f"FSA bottleneck grid from seg crop {tuple(cfg['segmentation']['crop_size'])}"
            )
        lhs = str(cfg.get("localization", {}).get("loc_head_select", "gt"))
        print(f"  Dual-head localization: True (head select: {lhs})")

    # --- Load checkpoint (weights only) ---
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ckpt_state = ckpt["model_state"]
        model_state = model.state_dict()
        # Partial load: matching keys only (e.g. dual_seg changes)
        matched, skipped = 0, 0
        for k in ckpt_state:
            if k in model_state and ckpt_state[k].shape == model_state[k].shape:
                model_state[k] = ckpt_state[k]
                matched += 1
            else:
                skipped += 1
        model.load_state_dict(model_state)
        if is_main:
            print(f"Loaded checkpoint: {args.checkpoint} (matched={matched}, skipped={skipped})")

    # --- Trainer ---
    trainer = LocSegTrainer(
        model, train_dataset, val_dataset, cfg, device=device,
        distributed=distributed, local_rank=local_rank, world_size=world_size,
    )

    # --- Train ---
    if args.resume:
        ckpt_meta = torch.load(args.resume, map_location="cpu")
        resume_phase = ckpt_meta.get("phase", 1)
        resume_epoch = ckpt_meta.get("epoch", 0)
        del ckpt_meta  # free before trainer.load_checkpoint reloads
        if is_main:
            print(f"\nResume: phase={resume_phase}, epoch={resume_epoch}")
            print(f"Checkpoint: {args.resume}")

        phase_fn = {
            1: trainer.train_phase1,
            2: trainer.train_phase2,
            3: trainer.train_phase3,
        }
        if resume_phase in phase_fn:
            phase_fn[resume_phase](resume_ckpt=args.resume)
        for p in range(resume_phase + 1, 4):
            if p == 2 and int(cfg["training"]["phase2"].get("epochs", 0)) <= 0:
                if is_main:
                    print("Skipping Phase 2 (epochs=0)")
                continue
            phase_fn[p]()

    elif args.only_phase > 0:
        phase_fn = {1: trainer.train_phase1, 2: trainer.train_phase2, 3: trainer.train_phase3}
        if args.only_phase in phase_fn:
            phase_fn[args.only_phase]()
        else:
            print(f"Invalid phase: {args.only_phase}")
    else:
        p2_epochs = int(cfg["training"]["phase2"].get("epochs", 0))
        if args.start_phase <= 1:
            trainer.train_phase1()
        if args.start_phase <= 2 and p2_epochs > 0:
            trainer.train_phase2()
        elif args.start_phase <= 2 and p2_epochs <= 0 and is_main:
            print("Skipping Phase 2 (training.phase2.epochs=0)")
        if args.start_phase <= 3:
            trainer.train_phase3()

    if is_main:
        print("\nTraining finished.")
    cleanup_distributed()


if __name__ == "__main__":
    main()
