import os
REPO_ROOT = os.environ.get('TN_ROOT', os.path.dirname(os.path.abspath(__file__)))
import copy
import json
import math
import os
import random
from collections import OrderedDict

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch import autocast
from torch import distributed as dist
from torch.utils.data import DataLoader
from tqdm.auto import trange, tqdm

from data.dataset import TNLocSegDataset, get_train_val_split, locseg_collate_fn
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context

from models.differentiable_crop import differentiable_crop_3d, hard_crop_3d
from models.loc_encoder import LocEncoder3D, generate_heatmap_target, soft_argmax_3d
from training.losses import LocalizationLoss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss (Abraham & Khan 2019), 针对极小前景的 dice 替代品.
        TI  = (TP + s) / (TP + alpha*FN + beta*FP + s)
        FTL = mean_c (1 - TI_c) ** gamma
    alpha>beta 偏向惩罚 FN (tiny FG 默认 0.7/0.3); gamma 1.0~1.5 进一步聚焦难样本.
    构造签名与 MemoryEfficientSoftDiceLoss 兼容, 可直接作为 DC_and_CE_loss.dice_class 的替身.
    """

    def __init__(
        self,
        apply_nonlin=None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1.0,
        ddp: bool = True,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 4.0 / 3.0,
    ):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = float(smooth)
        self.ddp = ddp
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            if x.shape == y.shape:
                y_onehot = y.to(torch.float32)
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            tp = (x * y_onehot).sum(axes, dtype=torch.float32)
            fp = (x * (1.0 - y_onehot)).sum(axes, dtype=torch.float32)
            fn = ((1.0 - x) * y_onehot).sum(axes, dtype=torch.float32)
        else:
            tp = (x * y_onehot * loss_mask).sum(axes, dtype=torch.float32)
            fp = (x * (1.0 - y_onehot) * loss_mask).sum(axes, dtype=torch.float32)
            fn = ((1.0 - x) * y_onehot * loss_mask).sum(axes, dtype=torch.float32)

        if self.batch_dice:
            if self.ddp:
                from nnunetv2.training.loss.dice import AllGatherGrad
                tp = AllGatherGrad.apply(tp).sum(0, dtype=torch.float32)
                fp = AllGatherGrad.apply(fp).sum(0, dtype=torch.float32)
                fn = AllGatherGrad.apply(fn).sum(0, dtype=torch.float32)
            tp = tp.sum(0, dtype=torch.float32)
            fp = fp.sum(0, dtype=torch.float32)
            fn = fn.sum(0, dtype=torch.float32)

        denom = (tp + self.alpha * fn + self.beta * fp + self.smooth).clamp_min(1e-8)
        tversky = (tp + self.smooth) / denom
        ftl = (1.0 - tversky).clamp_min(1e-8) ** self.gamma
        return ftl.mean()


class LegacyLocAugment:
    """旧定位策略的轻量增强: 小旋转/缩放 + 强度扰动 + 小模糊."""

    def __init__(
        self,
        affine_prob=0.5,
        max_rotate_deg=15.0,
        scale_min=0.9,
        scale_max=1.1,
        intensity_shift=0.1,
        intensity_scale=0.1,
        blur_prob=0.15,
    ):
        self.affine_prob = float(affine_prob)
        self.max_rotate_deg = float(max_rotate_deg)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.intensity_shift = float(intensity_shift)
        self.intensity_scale = float(intensity_scale)
        self.blur_prob = float(blur_prob)

    @staticmethod
    def _rot_x(a):
        c, s = torch.cos(a), torch.sin(a)
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float32)

    @staticmethod
    def _rot_y(a):
        c, s = torch.cos(a), torch.sin(a)
        return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float32)

    @staticmethod
    def _rot_z(a):
        c, s = torch.cos(a), torch.sin(a)
        return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

    def _affine_around_centroid(self, wb, centroid_norm, mask):
        if np.random.rand() >= self.affine_prob:
            return wb, centroid_norm, mask

        wb_t = torch.from_numpy(wb[None, None]).float()  # (1,1,D,H,W)
        d, h, w = wb.shape
        # centroid_norm 顺序是 (D,H,W), grid 需要 (x,y,z)=(W,H,D)
        c = torch.tensor(
            [2.0 * float(centroid_norm[2]) - 1.0, 2.0 * float(centroid_norm[1]) - 1.0, 2.0 * float(centroid_norm[0]) - 1.0],
            dtype=torch.float32,
        )

        max_rad = self.max_rotate_deg * np.pi / 180.0
        ax = torch.tensor(np.random.uniform(-max_rad, max_rad), dtype=torch.float32)
        ay = torch.tensor(np.random.uniform(-max_rad, max_rad), dtype=torch.float32)
        az = torch.tensor(np.random.uniform(-max_rad, max_rad), dtype=torch.float32)
        s = float(np.random.uniform(self.scale_min, self.scale_max))

        a = self._rot_z(az) @ self._rot_y(ay) @ self._rot_x(ax)
        a = a * s
        a_inv = torch.linalg.inv(a)
        t = c - (a_inv @ c)
        theta = torch.zeros((1, 3, 4), dtype=torch.float32)
        theta[0, :, :3] = a_inv
        theta[0, :, 3] = t

        grid = F.affine_grid(theta, size=(1, 1, d, h, w), align_corners=False)
        wb_out = F.grid_sample(wb_t, grid, mode="bilinear", padding_mode="border", align_corners=False)
        wb_np = wb_out[0, 0].cpu().numpy().astype(np.float32)

        if mask is not None:
            mask_t = torch.from_numpy(mask[None, None]).float()
            mask_out = F.grid_sample(mask_t, grid, mode="nearest", padding_mode="zeros", align_corners=False)
            mask_np = mask_out[0, 0].cpu().numpy().astype(mask.dtype, copy=False)
        else:
            mask_np = None

        # 围绕 centroid 做仿射, centroid 标签不变
        return wb_np, centroid_norm, mask_np

    def __call__(self, wb, centroid_norm, mask):
        wb = wb.copy()
        centroid_norm = centroid_norm.copy()
        mask_out = None if mask is None else mask.copy()

        wb, centroid_norm, mask_out = self._affine_around_centroid(wb, centroid_norm, mask_out)

        if self.intensity_scale > 0:
            scale = 1.0 + np.random.uniform(-self.intensity_scale, self.intensity_scale)
            wb = wb * scale
        if self.intensity_shift > 0:
            shift = np.random.uniform(-self.intensity_shift, self.intensity_shift)
            wb = wb + shift
        if self.blur_prob > 0 and np.random.rand() < self.blur_prob:
            wb_t = torch.from_numpy(wb[None, None]).float()
            wb = F.avg_pool3d(wb_t, kernel_size=3, stride=1, padding=1)[0, 0].cpu().numpy().astype(np.float32)

        return wb, centroid_norm, mask_out


def _to_tuple3(v):
    if isinstance(v, (tuple, list)):
        return (int(v[0]), int(v[1]), int(v[2]))
    i = int(v)
    return (i, i, i)


class LocSegWrapper(nn.Module):
    def __init__(self, seg_net: nn.Module, loc_net: nn.Module):
        super().__init__()
        self.seg_net = seg_net
        self.loc_net = loc_net

    def forward_loc(self, x):
        hm_left, hm_right = self.loc_net(x)
        c_left = soft_argmax_3d(hm_left)
        c_right = soft_argmax_3d(hm_right)
        return hm_left, hm_right, c_left, c_right

    def forward_seg(self, x):
        return self.seg_net(x)

    def forward(self, x):
        hm_left, hm_right, c_left, c_right = self.forward_loc(x)
        seg_out = self.forward_seg(x)
        return {
            "seg": seg_out,
            "heatmap_left": hm_left,
            "heatmap_right": hm_right,
            "centroid_left": c_left,
            "centroid_right": c_right,
        }


class nnUNetTrainerLoc(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.prepared_data_dir = os.environ.get("TN_PREPARED_DATA_DIR", os.path.join(REPO_ROOT, 'prepared_data'))
        loc_size_str = os.environ.get("TN_LOC_INPUT_SIZE", "128,128,128")
        self.loc_input_size = _to_tuple3([int(i) for i in loc_size_str.split(",")])
        self.phase1_epochs = int(os.environ.get("TN_PHASE1_EPOCHS", "150"))
        self.phase2_epochs = int(os.environ.get("TN_PHASE2_EPOCHS", "250"))
        self.phase3_epochs = int(os.environ.get("TN_PHASE3_EPOCHS", "600"))
        self.phase3_warmup_epochs = int(os.environ.get("TN_PHASE3_WARMUP_EPOCHS", "40"))
        self.phase3_min_gt_mix = float(os.environ.get("TN_PHASE3_MIN_GT_MIX", "0.2"))
        self.loc_loss_weight = float(os.environ.get("TN_LOC_LOSS_WEIGHT", "1.0"))
        _user_sigma = float(os.environ.get("TN_LOC_SIGMA", "3.5"))
        _ref_resolution = 96
        self.loc_sigma = _user_sigma * (self.loc_input_size[0] / _ref_resolution)
        self.loc_lr = float(os.environ.get("TN_LOC_LR", "1e-3"))
        self.loc_phase1_lr = float(os.environ.get("TN_LOC_PHASE1_LR", str(self.loc_lr)))
        self.loc_phase3_lr = float(os.environ.get("TN_LOC_PHASE3_LR", "5e-4"))
        self.loc_phase3_lr_schedule = os.environ.get("TN_LOC_PHASE3_LR_SCHEDULE", "constant")
        self.loc_phase3_lr_min_ratio = float(os.environ.get("TN_LOC_PHASE3_LR_MIN_RATIO", "0.1"))
        self.loc_phase1_weight_decay = float(os.environ.get("TN_LOC_PHASE1_WEIGHT_DECAY", "1e-5"))
        self.loc_phase1_lr_schedule = os.environ.get("TN_LOC_PHASE1_LR_SCHEDULE", "cosine")
        self.loc_phase1_lr_min_ratio = float(os.environ.get("TN_LOC_PHASE1_LR_MIN_RATIO", "0.01"))
        self.case_cache_size = int(os.environ.get("TN_CASE_CACHE_SIZE", "24"))
        self.loc_batch_size_phase1 = int(os.environ.get("TN_LOC_BS_PHASE1", "8"))
        self.seg_batch_size_override = int(os.environ.get("TN_SEG_BS", "0"))
        self.phase2_use_loc_crop = os.environ.get("TN_PHASE2_USE_LOC_CROP", "1") in ("1", "true", "True")
        self.phase2_precomputed_dir = os.environ.get("TN_PHASE2_PRECOMPUTED_DIR", "").strip()
        self.phase2_val_use_pred_crop = os.environ.get("TN_PHASE2_VAL_USE_PRED_CROP", "1") in ("1", "true", "True")
        self.phase2_val_wb_chunk_size = int(os.environ.get("TN_PHASE2_VAL_WB_CHUNK", "2"))
        self.phase2_vessel_oversample_prob = float(os.environ.get("TN_PHASE2_VESSEL_OVERSAMPLE_PROB", "0.0"))
        self.phase2_vessel_min_voxels = int(os.environ.get("TN_PHASE2_VESSEL_MIN_VOXELS", "1"))
        self.vessel_loss_weight = float(os.environ.get("TN_VESSEL_LOSS_WEIGHT", "0.0"))
        # phase3 whole-brain 路径与 phase2 预计算路径独立; 默认可单独开过采样 vessel case
        self.phase3_vessel_oversample_prob = float(os.environ.get("TN_PHASE3_VESSEL_OVERSAMPLE_PROB", "0.0"))
        # 验证时用 GT 质心裁 patch: 指标近似「固定 ROI baseline」，不包含 loc 漂移；部署仍应用 pred crop
        self.phase3_val_use_gt_crop = os.environ.get("TN_PHASE3_VAL_USE_GT_CROP", "0") in (
            "1",
            "true",
            "True",
        )
        self.phase3_fg_aux_weight = float(os.environ.get("TN_PHASE3_FG_AUX_WEIGHT", "0.35"))
        # phase3 分割支路 poly 初值: 未设则沿用 plans 的 initial_lr (常与 0.01)。
        # 当 phase1/2=0 直接 phase3 时, 0.01 与 loc(如 1e-3) 常差 10 倍, 可设为本变量与 loc 同量级。
        _p3s = os.environ.get("TN_PHASE3_SEG_LR", "").strip()
        self.phase3_seg_base_lr = float(_p3s) if _p3s else float(self.initial_lr)
        self.skip_bad_seg_loss = os.environ.get("TN_SKIP_BAD_SEG_LOSS", "1") in ("1", "true", "True")
        self.seg_loss_skip_threshold = float(os.environ.get("TN_SEG_LOSS_SKIP_THRESHOLD", "5.0"))
        # 主分割 loss 选择: "default" = DC+CE; "focal_tversky" = CE + Focal-Tversky
        # (针对极小 FG; alpha>beta 偏向召回, gamma>1 聚焦难样本)
        self.tn_loss_kind = os.environ.get("TN_LOSS", "default").strip().lower()
        self.ft_alpha = float(os.environ.get("TN_FT_ALPHA", "0.7"))
        self.ft_beta = float(os.environ.get("TN_FT_BETA", "0.3"))
        self.ft_gamma = float(os.environ.get("TN_FT_GAMMA", str(4.0 / 3.0)))
        self.tn_loss_ce_weight = float(os.environ.get("TN_LOSS_CE_WEIGHT", "0.5"))
        self.tn_loss_tversky_weight = float(os.environ.get("TN_LOSS_TVERSKY_WEIGHT", "1.0"))
        self.val_interval = int(os.environ.get("TN_VAL_INTERVAL", "5"))
        self.num_iterations_per_epoch = int(os.environ.get("TN_NUM_ITERS_PER_EPOCH", "100"))
        self.num_val_iterations_per_epoch = int(os.environ.get("TN_NUM_VAL_ITERS_PER_EPOCH", "50"))
        self.num_iterations_per_epoch_phase3 = int(
            os.environ.get("TN_NUM_ITERS_PER_EPOCH_PHASE3", str(self.num_iterations_per_epoch))
        )
        self.num_val_iterations_per_epoch_phase3 = int(
            os.environ.get("TN_NUM_VAL_ITERS_PER_EPOCH_PHASE3", str(self.num_val_iterations_per_epoch))
        )
        # phase3 val 全覆盖模式: 让每个 val case 平均被采到 ~cover_factor 次,
        # pseudo dice 标准差从 ±0.03 (30 iters) 降到 ±0.01 (4 patch/case × 75 case)
        self.phase3_val_full_coverage = os.environ.get("TN_PHASE3_VAL_FULL", "0") in ("1", "true", "True")
        self.phase3_val_coverage_factor = int(os.environ.get("TN_PHASE3_VAL_COVER", "4"))
        self._phase3_val_iters_cached = None  # do_split() 结果缓存, 避免每 val 调用重复 IO
        self.phase1_train_val_split = float(os.environ.get("TN_PHASE1_TRAIN_VAL_SPLIT", "0.8"))
        self.phase1_split_seed = int(os.environ.get("TN_PHASE1_SPLIT_SEED", "42"))
        self.phase1_num_workers = int(os.environ.get("TN_PHASE1_NUM_WORKERS", "4"))
        self.phase1_pin_memory = os.environ.get("TN_PHASE1_PIN_MEMORY", "1") in ("1", "true", "True")
        self.phase1_lr_flip_prob = float(os.environ.get("TN_PHASE1_LR_FLIP_PROB", "0.5"))
        self.phase1_affine_prob = float(os.environ.get("TN_PHASE1_AFFINE_PROB", "0.4"))
        self.phase1_max_rotate_deg = float(os.environ.get("TN_PHASE1_MAX_ROTATE_DEG", "15"))
        self.phase1_scale_min = float(os.environ.get("TN_PHASE1_SCALE_MIN", "0.9"))
        self.phase1_scale_max = float(os.environ.get("TN_PHASE1_SCALE_MAX", "1.1"))
        self.phase1_intensity_shift = float(os.environ.get("TN_PHASE1_INTENSITY_SHIFT", "0.05"))
        self.phase1_intensity_scale = float(os.environ.get("TN_PHASE1_INTENSITY_SCALE", "0.05"))
        self.phase1_blur_prob = float(os.environ.get("TN_PHASE1_BLUR_PROB", "0.0"))
        # None / "openneuro" / "inhouse" — 仅 phase1 old-loader 划分 train/val 时使用
        _pcf = os.environ.get("TN_PHASE1_CASE_FILTER", "").strip()
        self.phase1_case_filter = _pcf if _pcf and _pcf.lower() not in ("none", "all") else None

        self.smooth_sigma = float(os.environ.get("TN_SMOOTH_SIGMA", "0"))
        _ts = os.environ.get("TN_WB_TARGET_SPACING", "").strip()
        if _ts and _ts.lower() not in ("0", "native", "off", "none"):
            parts = [float(x) for x in _ts.replace(" ", "").split(",") if x]
            if len(parts) == 1:
                self.wb_target_spacing = (parts[0], parts[0], parts[0])
            elif len(parts) == 3:
                self.wb_target_spacing = tuple(parts)
            else:
                raise ValueError(
                    "TN_WB_TARGET_SPACING must be one isotropic value or three (sx,sy,sz) mm, "
                    f"got {_ts!r}"
                )
        else:
            self.wb_target_spacing = None
        self.wb_use_plans_fg_norm = os.environ.get("TN_WB_USE_PLANS_FG_NORM", "0") in (
            "1",
            "true",
            "True",
        )
        # 修 bug (b): 标准 nnU-Net CT preprocessing 在 z-score 之前会把强度 clip 到
        # foreground 的 [0.5%, 99.5%] 分位 (避免空气/骨/造影极端值进入网络).
        # 旧 wholebrain bridge 只做 z-score 不做 clip, 让 phase3 输入的 air/bone 范围
        # 远超 D096 训练分布 → seg net 在那些 voxel 上响应失控. 默认开启 clip 修复.
        self.wb_norm_clip = os.environ.get("TN_WB_NORM_CLIP", "1") in ("1", "true", "True")
        _ch0 = plans.get("foreground_intensity_properties_per_channel", {}).get("0", {})
        self._wb_plans_fg_mean = float(_ch0.get("mean", 0.0))
        self._wb_plans_fg_std = float(max(_ch0.get("std", 1.0), 1e-8))
        # plans 里 percentile_00_5 / 99_5 由 nnU-Net 预处理算 fg voxel 的 [0.5%, 99.5%] 得到
        self._wb_plans_fg_lo = float(_ch0.get("percentile_00_5", float("-inf")))
        self._wb_plans_fg_hi = float(_ch0.get("percentile_99_5", float("inf")))
        # 修 bug (c): differentiable_crop_3d 默认 padding_mode='zeros', 当 patch 触脑边界时
        # 会塞入 0 值, 在 z-score 后等价 mid-vessel HU 信号, 让网络在边界 voxel 上学到伪特征.
        # 改成 'border' 让边界外延拓边界值 (与 nnU-Net 自带 sliding window padding 同行为).
        self.wb_crop_padding_mode = os.environ.get("TN_WB_CROP_PADDING", "border").strip().lower()
        if self.wb_crop_padding_mode not in ("zeros", "border", "reflection"):
            raise ValueError(
                f"TN_WB_CROP_PADDING must be one of zeros/border/reflection, "
                f"got {self.wb_crop_padding_mode!r}"
            )
        self._wb_bridge_cfg_logged = False
        self.inner_dice_enable = os.environ.get("TN_INNER_DICE_ENABLE", "1") in ("1", "true", "True")
        self.inner_dice_crop_size = int(os.environ.get("TN_INNER_DICE_CROP", "48"))
        self.phase3_seg_aug_enabled = os.environ.get("TN_PHASE3_SEG_AUG", "1") in ("1", "true", "True")
        self._phase3_seg_transforms = None  # built lazily in _get_phase3_seg_transforms()
        # phase3 LOC 输入 (whole-brain) 的 augmentation. 默认关 (与 inhouse 行为兼容).
        # 小数据 fine-tune (e.g. OpenNeuro 64 case) 时强烈推荐开, 否则 loc 只背 train brain
        # 不学 invariant 特征. 旋转/缩放都围绕 centroid 进行 → centroid 在 norm 空间不变.
        self.loc_phase3_aug_enabled = os.environ.get("TN_LOC_PHASE3_AUG", "0") in ("1", "true", "True")
        self.loc_aug_affine_prob = float(os.environ.get("TN_LOC_AUG_AFFINE_PROB", "0.7"))
        self.loc_aug_max_rotate_deg = float(os.environ.get("TN_LOC_AUG_MAX_ROTATE_DEG", "15"))
        self.loc_aug_scale_min = float(os.environ.get("TN_LOC_AUG_SCALE_MIN", "0.9"))
        self.loc_aug_scale_max = float(os.environ.get("TN_LOC_AUG_SCALE_MAX", "1.1"))
        self.loc_aug_gamma_prob = float(os.environ.get("TN_LOC_AUG_GAMMA_PROB", "0.3"))
        self.loc_aug_gamma_min = float(os.environ.get("TN_LOC_AUG_GAMMA_MIN", "0.7"))
        self.loc_aug_gamma_max = float(os.environ.get("TN_LOC_AUG_GAMMA_MAX", "1.3"))

        self.num_epochs = self.phase1_epochs + self.phase2_epochs + self.phase3_epochs
        self.loc_channels = (16, 32, 64, 128, 256)
        self.loc_loss_fn = LocalizationLoss(
            heatmap_weight=float(os.environ.get("TN_LOC_HM_WEIGHT", "1.0")),
            centroid_weight=float(os.environ.get("TN_LOC_CENTROID_WEIGHT", "3.5")),
            sharpness_weight=float(os.environ.get("TN_LOC_SHARP_WEIGHT", "0.3")),
            heatmap_dice_weight=float(os.environ.get("TN_LOC_HM_DICE_WEIGHT", "0.3")),
        )

        self._case_mapping = {}
        self._wb_cache = OrderedDict()
        self._gt_seg_cache = OrderedDict()
        self._best_phase_metric = {"phase1": None, "phase2": None, "phase3": None}
        self._last_logged_phase = None
        self._freeze_state = None
        self._optimizer_built_for_phase = None
        self._phase1_train_loader = None
        self._phase1_val_loader = None
        self._skip_bad_seg_loss_count_epoch = 0
        self._phase2_crop_cache = OrderedDict()
        self._phase2_vessel_positive_keys = []

        self._init_case_mapping()
        self._init_phase2_vessel_keys()

        _o_fp = os.environ.get("TN_OVERSAMPLE_FOREGROUND_PERCENT", "").strip()
        if _o_fp:
            self.oversample_foreground_percent = float(_o_fp)
            self.print_to_log_file(
                f"[sampling] oversample_foreground_percent={self.oversample_foreground_percent} "
                f"(TN_OVERSAMPLE_FOREGROUND_PERCENT)"
            )

        if self.seg_batch_size_override > 0:
            old_bs = self.configuration_manager.batch_size
            self.configuration_manager.configuration["batch_size"] = int(self.seg_batch_size_override)
            self.batch_size = int(self.seg_batch_size_override)
            try:
                self._set_batch_size_and_oversample()
            except Exception:
                pass
            self.print_to_log_file(f"[seg-batch] override batch_size {old_bs} -> {self.configuration_manager.batch_size}")

    def _build_loss(self):
        """
        TN_LOSS=default       -> 父类 DC+CE (原行为)
        TN_LOSS=focal_tversky -> CE + FocalTversky, dice_class 直接换成 FocalTverskyLoss,
                                 走父类同一套 DeepSupervisionWrapper 逻辑.
        """
        if self.tn_loss_kind != "focal_tversky":
            return super()._build_loss()

        if self.label_manager.has_regions:
            # region-based 任务走 BCE 路径, 不在本任务范围; 退回父类避免静默错误
            self.print_to_log_file(
                "[loss] TN_LOSS=focal_tversky 不支持 region-based label, 回退到父类 DC+BCE"
            )
            return super()._build_loss()

        from functools import partial
        from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

        ft_class = partial(
            FocalTverskyLoss,
            alpha=self.ft_alpha,
            beta=self.ft_beta,
            gamma=self.ft_gamma,
        )
        loss = DC_and_CE_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {},
            weight_ce=self.tn_loss_ce_weight,
            weight_dice=self.tn_loss_tversky_weight,
            ignore_label=self.label_manager.ignore_label,
            dice_class=ft_class,
        )
        self.print_to_log_file(
            f"[loss] FocalTversky enabled  alpha={self.ft_alpha} beta={self.ft_beta} "
            f"gamma={self.ft_gamma:.4f}  weights: ce={self.tn_loss_ce_weight} "
            f"tversky={self.tn_loss_tversky_weight}"
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 1e-6 if (self.is_ddp and not self._do_i_compile()) else 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def configure_optimizers(self):
        mod = self.network.module if self.is_ddp else self.network
        if hasattr(mod, "_orig_mod"):
            mod = mod._orig_mod
        if not hasattr(mod, "loc_net") or not hasattr(mod, "seg_net"):
            return super().configure_optimizers()

        seg_params = [p for p in mod.seg_net.parameters() if p.requires_grad]
        loc_params = [p for p in mod.loc_net.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {"params": seg_params, "lr": self.initial_lr, "name": "seg", "weight_decay": self.weight_decay},
                {"params": loc_params, "lr": self.loc_lr, "name": "loc", "weight_decay": self.weight_decay},
            ],
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        self.print_to_log_file(
            f"[optimizer] seg_lr={self.initial_lr:.6g}, loc_lr={self.loc_lr:.6g}, weight_decay={self.weight_decay}"
        )
        return optimizer, lr_scheduler

    def _rebuild_optimizer_for_phase(self, phase: str):
        mod = self.network.module if self.is_ddp else self.network
        if hasattr(mod, "_orig_mod"):
            mod = mod._orig_mod
        seg_params = [p for p in mod.seg_net.parameters() if p.requires_grad]
        loc_params = [p for p in mod.loc_net.parameters() if p.requires_grad]

        if phase == "phase1":
            param_groups = [
                {
                    "params": loc_params,
                    "lr": self.loc_phase1_lr,
                    "name": "loc",
                    "weight_decay": self.loc_phase1_weight_decay,
                }
            ]
            base_lr = self.loc_phase1_lr
        elif phase == "phase2":
            param_groups = [
                {
                    "params": seg_params,
                    "lr": self.initial_lr,
                    "name": "seg",
                    "weight_decay": self.weight_decay,
                }
            ]
            base_lr = self.initial_lr
        else:
            param_groups = [
                {
                    "params": seg_params,
                    "lr": self.phase3_seg_base_lr,
                    "name": "seg",
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": loc_params,
                    "lr": self.loc_phase3_lr,
                    "name": "loc",
                    "weight_decay": self.weight_decay,
                },
            ]
            # Poly 衰减以 phase3 分割初值为基准, 与 loc 分支无关 (随后会单独写回 loc 学习率)
            base_lr = self.phase3_seg_base_lr

        self.optimizer = torch.optim.SGD(
            param_groups,
            lr=base_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        self.lr_scheduler = PolyLRScheduler(self.optimizer, base_lr, self.num_epochs)
        self._optimizer_built_for_phase = phase
        self.print_to_log_file(
            f"[optimizer-phase] rebuilt for {phase}: groups={[pg.get('name','?') for pg in self.optimizer.param_groups]}"
        )
        if phase == "phase3" and self.phase3_seg_base_lr != float(self.initial_lr):
            self.print_to_log_file(
                f"[phase3] seg poly base_lr={self.phase3_seg_base_lr} (plans initial_lr={float(self.initial_lr):.6g}, "
                f"set TN_PHASE3_SEG_LR to override)"
            )

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        arch_kw = copy.deepcopy(arch_init_kwargs)
        _dp = os.environ.get("TN_SEG_DROPOUT_P", "").strip()
        if _dp:
            p = float(_dp)
            arch_kw["dropout_op"] = "torch.nn.Dropout3d"
            arch_kw["dropout_op_kwargs"] = {"p": p, "inplace": True}
            print(f"[build_network] seg PlainConvUNet Dropout3d p={p} (TN_SEG_DROPOUT_P)")

        seg_net = get_network_from_plans(
            architecture_class_name,
            arch_kw,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        # --- 可选: 为 seg UNet 注入 SE 通道注意力 ---
        seg_se = os.environ.get("TN_SEG_SE", "0").strip()
        if seg_se and seg_se not in ("0", "false", "no"):
            from models.seg_se_wrapper import SEPlainConvUNet
            se_reduction = int(os.environ.get("TN_SEG_SE_REDUCTION", "16"))
            seg_net = SEPlainConvUNet(seg_net, se_reduction=se_reduction)
            print(f"[build_network] seg_net wrapped with SE (reduction={se_reduction})")

        # 选择 loc encoder 架构
        #   attn (默认): LocEncoder3D with spatial AttentionGate3D
        #   se: LocEncoderSE3D with SE channel attention (2025 J Neurosurg SOTA)
        loc_arch = os.environ.get("TN_LOC_ARCH", "attn").lower()
        if loc_arch == "se":
            from models.loc_encoder_se import LocEncoderSE3D
            loc_net = LocEncoderSE3D(in_channels=num_input_channels, channels=(16, 32, 64, 128, 256))
            print(f"[build_network] loc_arch=se → LocEncoderSE3D (SE channel attention)")
        else:
            loc_net = LocEncoder3D(in_channels=num_input_channels, channels=(16, 32, 64, 128, 256))
            print(f"[build_network] loc_arch=attn → LocEncoder3D (AttentionGate3D spatial attention)")
        return LocSegWrapper(seg_net=seg_net, loc_net=loc_net)

    def _init_case_mapping(self):
        if nnUNet_raw is None:
            return
        mapping_file = os.path.join(nnUNet_raw, self.plans_manager.dataset_name, "case_mapping.json")
        if not os.path.isfile(mapping_file):
            self.print_to_log_file(f"[wholebrain-bridge] case mapping not found: {mapping_file}")
            return
        with open(mapping_file) as f:
            self._case_mapping = json.load(f)
        self.print_to_log_file(f"[wholebrain-bridge] loaded case mapping: {len(self._case_mapping)} entries")

    def _get_phase_name(self):
        e = self.current_epoch
        if e < self.phase1_epochs:
            return "phase1"
        if e < self.phase1_epochs + self.phase2_epochs:
            return "phase2"
        return "phase3"

    def _apply_freeze_for_phase(self, phase: str):
        if self._freeze_state == phase:
            return
        mod = self.network.module if self.is_ddp else self.network
        if phase == "phase1":
            for p in mod.seg_net.parameters():
                p.requires_grad = False
            for p in mod.loc_net.parameters():
                p.requires_grad = True
        elif phase == "phase2":
            for p in mod.seg_net.parameters():
                p.requires_grad = True
            for p in mod.loc_net.parameters():
                p.requires_grad = False
        else:
            for p in mod.seg_net.parameters():
                p.requires_grad = True
            for p in mod.loc_net.parameters():
                p.requires_grad = True
        self._freeze_state = phase

    @staticmethod
    def _gt_left_mask_padded(gt_centroid_norm, wb_shapes, padded_shape):
        """
        判断 GT 质心是否落在该 case 的 d-轴左半 (case_d/2 之内).
        gt_centroid_norm: (B, 3), 在 padded 张量 [0,1] 坐标系下;
        wb_shapes: (B, 3) per-case 实际 shape (D,H,W);
        padded_shape: tuple/torch.Size, wb_tensor.shape[2:] 即 (max_D, max_H, max_W).
        返回: (B,) bool tensor.
        """
        pad_d_minus1 = max(int(padded_shape[0]) - 1, 1)
        gt_voxel_d = gt_centroid_norm[:, 0].float() * float(pad_d_minus1)
        half_case_d = wb_shapes[:, 0].float() * 0.5
        return gt_voxel_d < half_case_d

    def _select_centroid_by_gt_side(self, c_left, c_right, gt_centroid_norm,
                                    wb_shapes=None, padded_shape=None):
        # 兼容老调用 (无 wb_shapes/padded_shape): 直接按 [0,1] 0.5 划分;
        # 新路径 (build_whole_brain_batch 返回 padded-norm centroid): 用 case shape 校正左右。
        if wb_shapes is None or padded_shape is None:
            use_left = (gt_centroid_norm[:, 0] < 0.5).unsqueeze(1)
        else:
            use_left = self._gt_left_mask_padded(gt_centroid_norm, wb_shapes, padded_shape).unsqueeze(1)
        return torch.where(use_left, c_left, c_right)

    def set_deep_supervision_enabled(self, enabled: bool):
        mod = self.network.module if self.is_ddp else self.network
        if hasattr(mod, "seg_net") and hasattr(mod.seg_net, "decoder"):
            mod.seg_net.decoder.deep_supervision = enabled
        elif hasattr(mod, "decoder"):
            mod.decoder.deep_supervision = enabled
        else:
            raise AttributeError("Cannot set deep supervision on current network")

    @staticmethod
    def _wb_resampled_shape(native_shape: tuple[int, ...], spacing_mm, target_spacing_mm):
        """native_shape × spacing → 目标各向同性/目标 spacing 下的体素尺寸 (与 nnU-Net round 约定一致)。"""
        native_shape = np.array(native_shape, dtype=np.float64)[:3]
        spacing_mm = np.array(spacing_mm, dtype=np.float64)[:3]
        target_spacing_mm = np.array(target_spacing_mm, dtype=np.float64)[:3]
        out = np.maximum(1, np.round(native_shape * spacing_mm / target_spacing_mm).astype(np.int64))
        return tuple(int(x) for x in out)

    @staticmethod
    def _resample_vol_trilinear(vol: np.ndarray, out_shape: tuple[int, int, int]) -> np.ndarray:
        t = torch.from_numpy(vol.astype(np.float32))[None, None]
        out = F.interpolate(t, size=out_shape, mode="trilinear", align_corners=False)
        return out[0, 0].cpu().numpy().astype(np.float32)

    @staticmethod
    def _resample_seg_nearest(seg: np.ndarray, out_shape: tuple[int, int, int]) -> np.ndarray:
        t = torch.from_numpy(seg.astype(np.float32))[None, None]
        out = F.interpolate(t, size=out_shape, mode="nearest")
        return torch.round(out[0, 0]).long().cpu().numpy().astype(np.int64)

    def _load_case_whole_brain(self, nnunet_key: str):
        if nnunet_key in self._wb_cache:
            item = self._wb_cache.pop(nnunet_key)
            self._wb_cache[nnunet_key] = item
            return item

        prepared_case = self._case_mapping.get(nnunet_key, None)
        if prepared_case is None:
            raise KeyError(f"missing case mapping for {nnunet_key}")

        case_dir = os.path.join(self.prepared_data_dir, prepared_case)
        info_file = os.path.join(case_dir, "info.json")
        centroid_file = os.path.join(case_dir, "centroid.npy")
        if not os.path.isfile(info_file):
            raise FileNotFoundError(f"missing info file: {info_file}")
        if not os.path.isfile(centroid_file):
            raise FileNotFoundError(f"missing centroid file: {centroid_file}")

        with open(info_file) as f:
            info = json.load(f)

        if not self._wb_bridge_cfg_logged:
            self._wb_bridge_cfg_logged = True
            self.print_to_log_file(
                f"[wholebrain-bridge] TN_WB_TARGET_SPACING={self.wb_target_spacing!r}, "
                f"TN_WB_USE_PLANS_FG_NORM={self.wb_use_plans_fg_norm} "
                f"(plans_fg mean={self._wb_plans_fg_mean:.2f}, std={self._wb_plans_fg_std:.2f}); "
                f"TN_WB_NORM_CLIP={self.wb_norm_clip} "
                f"(plans_fg [p0.5, p99.5]=[{self._wb_plans_fg_lo:.1f}, {self._wb_plans_fg_hi:.1f}]); "
                f"TN_WB_CROP_PADDING={self.wb_crop_padding_mode}"
            )

        wb_nii = nib.load(info["nii_path"])
        wb = wb_nii.get_fdata().astype(np.float32)
        if wb.ndim == 4:
            wb = wb[..., 0]
        native_shape = tuple(int(x) for x in wb.shape[:3])
        spacing_prep = np.array(wb_nii.header.get_zooms()[:3], dtype=np.float64)

        gt_centroid = np.load(centroid_file).astype(np.float32)
        if self.wb_target_spacing is not None:
            out_shape = self._wb_resampled_shape(native_shape, spacing_prep, self.wb_target_spacing)
            wb = self._resample_vol_trilinear(wb, out_shape)
            tgt_sp = np.array(self.wb_target_spacing, dtype=np.float64)
            gt_centroid = gt_centroid * (spacing_prep / tgt_sp)

        # 修 bug (#24, 2026-05-07): 之前默认走 plans-fg-norm 是 CTNormalization 的逻辑,
        # 但 D001 是 MRI (channel_names.0=='MRI'), plans 里 normalization_schemes='ZScoreNormalization'
        # use_mask_for_norm=False, 即 D096 stock nnUNet 用的是「per-case 全图 z-score, 不 clip」.
        # 强行套全局 plans-fg mean/std + clip → 给网络的 input std 只有 0.19 (target 1.0),
        # 56% 背景 voxel 被 clip 到同一常数 → 单 case 都过拟合不了.
        if self.wb_use_plans_fg_norm:
            # CTNormalization 路径 (CT 数据集才该开). 默认关闭, 仅留为可选.
            if self.wb_norm_clip and np.isfinite(self._wb_plans_fg_lo) and np.isfinite(self._wb_plans_fg_hi):
                wb = np.clip(wb, self._wb_plans_fg_lo, self._wb_plans_fg_hi)
            wb = (wb - self._wb_plans_fg_mean) / (self._wb_plans_fg_std + 1e-8)
        else:
            # ZScoreNormalization 路径 (与 D096 stock 完全一致): 整个 volume 全图 mean/std,
            # 不用 fg mask. 对齐 plans 里 use_mask_for_norm=False.
            mean = float(wb.mean())
            std = float(wb.std())
            wb = (wb - mean) / (std + 1e-8)

        if self.smooth_sigma > 0:
            wb = gaussian_filter(wb, sigma=self.smooth_sigma).astype(np.float32)

        wb_shape = np.array(wb.shape[:3], dtype=np.float32)
        gt_centroid_norm = gt_centroid / np.maximum(wb_shape - 1.0, 1.0)
        item = (wb, wb_shape, gt_centroid_norm.astype(np.float32))

        self._wb_cache[nnunet_key] = item
        if len(self._wb_cache) > self.case_cache_size:
            self._wb_cache.popitem(last=False)
        return item

    def _build_whole_brain_batch(self, keys):
        wb_list, shape_list, centroid_list = [], [], []
        max_d, max_h, max_w = 0, 0, 0
        for k in keys:
            wb, wb_shape, gt_centroid_norm = self._load_case_whole_brain(k)
            wb_list.append(wb)
            shape_list.append(wb_shape)
            centroid_list.append(gt_centroid_norm)
            d, h, w = wb.shape
            max_d = max(max_d, d)
            max_h = max(max_h, h)
            max_w = max(max_w, w)

        batch = np.zeros((len(keys), 1, max_d, max_h, max_w), dtype=np.float32)
        for i, wb in enumerate(wb_list):
            d, h, w = wb.shape
            batch[i, 0, :d, :h, :w] = wb

        # 关键修正: gt_centroid_norm 来自 _load_case_whole_brain 是 per-case 归一化;
        # batch 把不同尺寸的 wb 零填充到 (max_d,max_h,max_w), pred 走 _loc_forward 后是
        # padded [0,1] 空间; 必须把 GT 同步换到 padded [0,1], 否则下游 crop / heatmap target /
        # phase3 mix / loc 误差全部在两套坐标系里混算 -> Dice≈0、val_fg_stats pred>>gt、不重叠。
        case_shapes_arr = np.stack(shape_list).astype(np.float32)  # (B,3)
        per_case_norm = np.stack(centroid_list).astype(np.float32)  # (B,3)
        pad_shape_arr = np.array([max_d, max_h, max_w], dtype=np.float32)
        gt_voxel_in_case = per_case_norm * np.maximum(case_shapes_arr - 1.0, 1.0)
        gt_padded_norm = gt_voxel_in_case / np.maximum(pad_shape_arr - 1.0, 1.0)

        wb_tensor = torch.from_numpy(batch).to(self.device, non_blocking=True)
        wb_shapes = torch.from_numpy(case_shapes_arr).to(self.device, non_blocking=True)
        gt_centroids = torch.from_numpy(gt_padded_norm.astype(np.float32)).to(
            self.device, non_blocking=True
        )
        return wb_tensor, wb_shapes, gt_centroids

    def _max_dataset_label_value(self) -> int:
        labels = self.dataset_json.get("labels", {})
        if not labels:
            return 2
        return max(int(v) for v in labels.values())

    def reset_segmentation_heads(self, bg_bias: float = 0.0):
        """
        仅重置分割输出头（out_channels==num_classes 的 Conv3d），用于从塌缩 checkpoint 恢复。
        """
        mod = self.network.module if self.is_ddp else self.network
        if hasattr(mod, "_orig_mod"):
            mod = mod._orig_mod
        if not hasattr(mod, "seg_net"):
            self.print_to_log_file("[reset-seg-head] skipped: no seg_net")
            return
        num_classes = self._max_dataset_label_value() + 1
        reset_cnt = 0
        for m in mod.seg_net.modules():
            if isinstance(m, nn.Conv3d) and m.out_channels == num_classes:
                nn.init.kaiming_normal_(m.weight, a=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    # 背景偏置可选设为更小，减轻全背景先验
                    if m.bias.numel() > 0:
                        m.bias.data[0] = float(bg_bias)
                reset_cnt += 1
        self.print_to_log_file(
            f"[reset-seg-head] reset_conv3d={reset_cnt}, num_classes={num_classes}, bg_bias={bg_bias}"
        )

    def _load_gt_seg_resized_to_wb(self, nnunet_key: str, wb_shape_dhw: tuple[int, int, int]):
        """
        加载 case 的 GT 标签并按 affine 正确放回 prepared whole-brain 体素网格。

        nnU-Net 的 gt_segmentations 是 64³ ROI（各向同性 1mm，affine=I），不是全脑标签；
        而 prepared whole-brain 是原始 NIFTI（如 297×384×384，0.6mm 各向异性）。
        若像旧版直接 F.interpolate(64³ → wb_shape) 会忽略 spacing/affine，
        把 ROI 拉伸覆盖整个全脑，监督源被错放 ~50mm，phase3 的 seg 头永远不收敛。

        正确路径：
          1) 读 nnU-Net 64³ GT 与其 spacing；
          2) 读 prepared 全脑 spacing 与该 case 的 ROI 中心 (info.center_csv)；
          3) 把 64³ × nnu_spacing 物理空间用 nearest 重采样到 prepared spacing 的 voxel 网格
             (target = round(64 × nnu_spacing / prep_spacing))；
          4) 在全脑零张量中以 center_csv 为中心嵌入这块标签（边界自动裁剪）。

        修复后单 case 重心误差 ≤ 1 voxel（vs 旧版 ~90 voxel）。
        """
        cache_key = (nnunet_key, wb_shape_dhw)
        if cache_key in self._gt_seg_cache:
            item = self._gt_seg_cache.pop(cache_key)
            self._gt_seg_cache[cache_key] = item
            return item

        base = self.preprocessed_dataset_folder_base
        if not base:
            raise RuntimeError("preprocessed_dataset_folder_base is None; cannot load gt_segmentations")
        seg_path = os.path.join(base, "gt_segmentations", f"{nnunet_key}.nii.gz")
        if not os.path.isfile(seg_path):
            raise FileNotFoundError(f"Missing nnU-Net GT segmentation: {seg_path}")

        prepared_case = self._case_mapping.get(nnunet_key, None)
        if prepared_case is None:
            raise KeyError(
                f"missing case mapping for {nnunet_key}; cannot align GT to whole-brain"
            )
        case_dir = os.path.join(self.prepared_data_dir, prepared_case)
        info_path = os.path.join(case_dir, "info.json")
        centroid_path = os.path.join(case_dir, "centroid.npy")
        if not os.path.isfile(info_path):
            raise FileNotFoundError(f"missing prepared info: {info_path}")
        if not os.path.isfile(centroid_path):
            raise FileNotFoundError(f"missing prepared centroid: {centroid_path}")
        with open(info_path) as f:
            info = json.load(f)

        seg_nii = nib.load(seg_path)
        seg = seg_nii.get_fdata()
        if seg.ndim == 4:
            seg = seg[..., 0]
        seg = np.rint(seg).astype(np.int64)
        spacing_nnu = np.array(seg_nii.header.get_zooms()[:3], dtype=np.float64)

        wb_nii = nib.load(info["nii_path"])
        spacing_prep = np.array(wb_nii.header.get_zooms()[:3], dtype=np.float64)
        native_shape = tuple(int(x) for x in wb_nii.shape[:3])
        if self.wb_target_spacing is not None:
            expected_shape = self._wb_resampled_shape(native_shape, spacing_prep, self.wb_target_spacing)
        else:
            expected_shape = native_shape
        if tuple(int(x) for x in wb_shape_dhw) != tuple(int(x) for x in expected_shape):
            raise RuntimeError(
                f"wb_shape mismatch for {nnunet_key}: after WB resampling expect {expected_shape}, "
                f"got requested {wb_shape_dhw}"
            )

        # 64³ × nnu_spacing → **native** prepared_spacing 网格下应占的 voxel 数
        target_shape = np.maximum(
            1, np.round(np.array(seg.shape) * spacing_nnu / spacing_prep).astype(np.int64)
        )

        seg_t = torch.from_numpy(seg.astype(np.float32)).view(1, 1, *seg.shape)
        seg_t = F.interpolate(
            seg_t, size=tuple(int(x) for x in target_shape), mode="nearest"
        )
        seg_resamp = torch.round(seg_t[0, 0]).long().cpu().numpy()

        # 用 centroid.npy (= info.center_ras, 即 RAS voxel 索引), 与 _load_case_whole_brain
        # 的 loc 监督源完全一致。注意: info.center_csv 是医学查看器的原始坐标 (L→R/A→P/S→I 约定),
        # 不是 voxel 索引 — 用它做嵌入会有 ~5mm 的系统性偏移。
        center = np.round(np.load(centroid_path).astype(np.float64)).astype(np.int64)
        half = target_shape // 2

        full_native = np.zeros(native_shape, dtype=np.int64)

        def _emb_slice(c, h, t, dim):
            lo_full = c - h
            hi_full = lo_full + t
            lo_clip = max(0, int(lo_full))
            hi_clip = min(int(dim), int(hi_full))
            if hi_clip <= lo_clip:
                return None, None
            src_lo = lo_clip - int(lo_full)
            src_hi = src_lo + (hi_clip - lo_clip)
            return slice(lo_clip, hi_clip), slice(src_lo, src_hi)

        s0, ss0 = _emb_slice(center[0], half[0], target_shape[0], native_shape[0])
        s1, ss1 = _emb_slice(center[1], half[1], target_shape[1], native_shape[1])
        s2, ss2 = _emb_slice(center[2], half[2], target_shape[2], native_shape[2])
        if s0 is not None and s1 is not None and s2 is not None:
            full_native[s0, s1, s2] = seg_resamp[ss0, ss1, ss2]

        if native_shape == tuple(int(x) for x in wb_shape_dhw):
            full = full_native
        else:
            full = self._resample_seg_nearest(full_native, tuple(int(x) for x in wb_shape_dhw))

        mx = self._max_dataset_label_value()
        full = np.clip(full, 0, mx).astype(np.int64, copy=False)

        self._gt_seg_cache[cache_key] = full
        if len(self._gt_seg_cache) > max(1, self.case_cache_size) * 2:
            self._gt_seg_cache.popitem(last=False)
        return full

    def _subsample_keys_for_phase1_loc(self, keys):
        if self.loc_batch_size_phase1 <= 0:
            return keys
        if len(keys) <= self.loc_batch_size_phase1:
            return keys
        idx = torch.randperm(len(keys))[:self.loc_batch_size_phase1].tolist()
        return [keys[i] for i in idx]

    def _loc_forward(self, whole_brain_tensor, mod):
        loc_in = whole_brain_tensor
        if tuple(whole_brain_tensor.shape[2:]) != self.loc_input_size:
            loc_in = torch.nn.functional.interpolate(
                whole_brain_tensor, size=self.loc_input_size, mode="trilinear", align_corners=False
            )
        hm_l, hm_r, c_l, c_r = mod.forward_loc(loc_in)
        return hm_l, hm_r, c_l, c_r

    def _apply_loc_phase3_augmentation(self, wb_tensor, gt_centroid_norm):
        """
        给 phase3 train 时的 wb_tensor 加随机 affine + intensity aug, 强迫 loc 学 invariant 特征.
        旋转/缩放都围绕 GT centroid (centroid 是 affine 的 fixed point), 所以 centroid_norm 不变.
        不做 mirror — loc 网络有独立 left/right head, 翻转会破坏其语义.

        参数:
          wb_tensor: (B, 1, D, H, W), 已 z-score 归一化
          gt_centroid_norm: (B, 3) in [0, 1] (D, H, W ordering)
        返回: 增强后的 wb_tensor (centroid 不变, 不返回)
        """
        if not self.loc_phase3_aug_enabled:
            return wb_tensor
        B = wb_tensor.shape[0]
        device = wb_tensor.device

        # 1. Per-sample affine (rotation around centroid + scale)
        do_affine = (torch.rand(B, device=device) < self.loc_aug_affine_prob)
        if do_affine.any():
            theta_list = []
            for b in range(B):
                if not do_affine[b].item():
                    theta_list.append(torch.eye(3, 4, device=device, dtype=torch.float32))
                    continue
                angles = (np.random.rand(3) - 0.5) * 2 * self.loc_aug_max_rotate_deg * np.pi / 180
                scale = float(np.random.uniform(self.loc_aug_scale_min, self.loc_aug_scale_max))
                ax, ay, az = angles
                cx, sx = np.cos(ax), np.sin(ax)
                cy, sy = np.cos(ay), np.sin(ay)
                cz, sz = np.cos(az), np.sin(az)
                Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
                Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
                R = Rz @ Ry @ Rx
                # backward affine for grid_sample: M_inv = (1/scale) * R^-1
                M_inv = (1.0 / scale) * np.linalg.inv(R)
                # F.affine_grid theta uses (W, H, D) ordering for the last dim (xyz)
                # gt_centroid_norm[b] is (cd, ch, cw); convert to grid xyz = (2*cw-1, 2*ch-1, 2*cd-1)
                cd, ch, cw = gt_centroid_norm[b].detach().cpu().numpy()
                c_xyz = np.array([2 * cw - 1, 2 * ch - 1, 2 * cd - 1], dtype=np.float64)
                bias = c_xyz - M_inv @ c_xyz  # rotation around centroid → bias = c - M @ c
                theta_b = np.zeros((3, 4), dtype=np.float32)
                theta_b[:3, :3] = M_inv
                theta_b[:, 3] = bias
                theta_list.append(torch.from_numpy(theta_b).to(device, dtype=torch.float32))
            theta = torch.stack(theta_list, dim=0)
            grid = F.affine_grid(theta, wb_tensor.shape, align_corners=True)
            wb_tensor = F.grid_sample(
                wb_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True
            )

        # 2. Per-sample gamma intensity aug
        do_gamma = (torch.rand(B, device=device) < self.loc_aug_gamma_prob)
        if do_gamma.any():
            for b in range(B):
                if not do_gamma[b].item():
                    continue
                gamma = float(np.random.uniform(self.loc_aug_gamma_min, self.loc_aug_gamma_max))
                wb_min = wb_tensor[b].min()
                wb_max = wb_tensor[b].max()
                denom = (wb_max - wb_min).clamp_min(1e-8)
                wb_norm = (wb_tensor[b] - wb_min) / denom
                wb_tensor[b] = wb_norm.clamp_min(1e-8).pow(gamma) * denom + wb_min

        return wb_tensor

    def _crop_with_centroid(self, whole_brain_tensor, centroid_norm, patch_size):
        # 修 bug (c): 默认 padding_mode 由 'zeros' 改为 'border'.
        # zeros 在脑边界外塞入 0, 经 z-score 后等价 mid-vessel HU 信号 (~527 HU),
        # 让 seg net 在边界 voxel 上学到伪特征. border 延拓边界值, 与 nnU-Net
        # 自带 sliding window padding 行为一致. env TN_WB_CROP_PADDING 可改回.
        return differentiable_crop_3d(
            whole_brain_tensor, centroid_norm, patch_size,
            mode="bilinear", padding_mode=self.wb_crop_padding_mode,
        )

    def _load_and_crop_gt_label_patch(
        self,
        phase_keys,
        wb_shapes,
        padded_shape,
        centroid_norm,
        patch_size,
    ):
        """
        在 numpy/CPU 端按 centroid 直接裁出 (B,1,patch_d,patch_h,patch_w) 标签 patch 后再上 GPU,
        替代 "GPU 整脑 long 标签 + hard_crop_3d" 的高显存路径(大 case 下整脑 long 张量可达 45+ GiB → OOM)。

        语义与旧 _build_whole_brain_gt_label_tensor + _crop_label_with_centroid 完全等价:
          1. centroid_norm 在 padded [0,1] 坐标系(与 wb_tensor 同),padded voxel 中心 =
             round(centroid * (padded_shape - 1)) 后 clamp,与 grid_sample(align_corners=True) 边角对齐一致。
          2. 复刻 hard_crop_3d 的窗口边界推回:
                lo = max(0, c - half); hi = min(dim, lo + cs); lo = max(0, hi - cs)
             padded 边界外保持 0(等价于原版整脑 long zeros 填充 + slice)。
          3. case 在 padded 内左上角对齐 (batch[i,0,:d_i,:h_i,:w_i] = wb),所以 padded 内
             [lo:hi] 越过 case 实际边界的部分应为 0(GT 在 case 外不存在)。

        参数:
            phase_keys: list[str], 长度为 B
            wb_shapes:  (B,3) torch.Tensor 或 ndarray, 各 case 真实形状
            padded_shape: 三元组 (D,H,W), = wb_tensor.shape[2:]
            centroid_norm: (B,3) torch.Tensor on self.device, padded [0,1] 坐标
            patch_size: int 或 (d,h,w)

        返回: (B,1,crop_d,crop_h,crop_w) torch.long on self.device
        """
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        crop_d = int(patch_size[0])
        crop_h = int(patch_size[1])
        crop_w = int(patch_size[2])
        halves = (crop_d // 2, crop_h // 2, crop_w // 2)
        crops = (crop_d, crop_h, crop_w)
        pad_dims = (int(padded_shape[0]), int(padded_shape[1]), int(padded_shape[2]))

        centroid_np = centroid_norm.detach().to(torch.float32).cpu().numpy()
        if isinstance(wb_shapes, torch.Tensor):
            case_shapes_np = wb_shapes.detach().cpu().to(torch.int64).numpy()
        else:
            case_shapes_np = np.asarray(wb_shapes, dtype=np.int64)

        scale = np.array(
            [max(pad_dims[0] - 1, 1), max(pad_dims[1] - 1, 1), max(pad_dims[2] - 1, 1)],
            dtype=np.float32,
        )
        center_vox = np.rint(centroid_np * scale).astype(np.int64)
        center_vox[:, 0] = np.clip(center_vox[:, 0], 0, pad_dims[0] - 1)
        center_vox[:, 1] = np.clip(center_vox[:, 1], 0, pad_dims[1] - 1)
        center_vox[:, 2] = np.clip(center_vox[:, 2], 0, pad_dims[2] - 1)

        mx = self._max_dataset_label_value()
        use_uint8 = mx <= 255
        out_dtype = np.uint8 if use_uint8 else np.int64
        B = len(phase_keys)
        out = np.zeros((B, 1, crop_d, crop_h, crop_w), dtype=out_dtype)

        for i, k in enumerate(phase_keys):
            d_i = int(case_shapes_np[i, 0])
            h_i = int(case_shapes_np[i, 1])
            w_i = int(case_shapes_np[i, 2])
            case_dims = (d_i, h_i, w_i)

            # 修 bug (#25): 旧版用 shift-to-fit (`lo = max(0, hi-cs)`),
            # 越界时整个窗口往内挪 → centroid 在 GT patch 里不在 k=cs/2 处.
            # 但 image crop 用 grid_sample padding_mode='border', centroid 永远在
            # image patch 的几何中心 → GT/image 错位 (TN_0004 dim 1 错 24 voxel,
            # 网络永远学不会). 改成"以 centroid 为中心硬裁 + OOB 填 0", 与 image
            # 完全对齐. 边界外 GT 不存在, 用 0 (background) 填是正确语义.
            lo_raw = [
                int(center_vox[i, 0]) - halves[0],
                int(center_vox[i, 1]) - halves[1],
                int(center_vox[i, 2]) - halves[2],
            ]
            hi_raw = [lo_raw[d] + crops[d] for d in range(3)]

            seg_np = self._load_gt_seg_resized_to_wb(k, case_dims)
            # crop intersection in case-frame
            d_lo = max(0, lo_raw[0]); d_hi = min(d_i, hi_raw[0])
            h_lo = max(0, lo_raw[1]); h_hi = min(h_i, hi_raw[1])
            w_lo = max(0, lo_raw[2]); w_hi = min(w_i, hi_raw[2])
            if d_hi <= d_lo or h_hi <= h_lo or w_hi <= w_lo:
                continue  # 整个 patch 都在 case 外, 留 zeros (out 已是 0)
            # 在输出 patch 内的目标偏移
            pd_lo = d_lo - lo_raw[0]; pd_hi = pd_lo + (d_hi - d_lo)
            ph_lo = h_lo - lo_raw[1]; ph_hi = ph_lo + (h_hi - h_lo)
            pw_lo = w_lo - lo_raw[2]; pw_hi = pw_lo + (w_hi - w_lo)

            src = seg_np[d_lo:d_hi, h_lo:h_hi, w_lo:w_hi]
            if use_uint8:
                src = np.clip(src, 0, 255).astype(np.uint8, copy=False)
            out[i, 0, pd_lo:pd_hi, ph_lo:ph_hi, pw_lo:pw_hi] = src

        out_t = torch.from_numpy(out).to(self.device, non_blocking=True).long()
        return out_t

    def _crop_around_target_foreground_centroid(self, output, target_eval, crop_size: int):
        """
        在当前 seg patch 内，以 GT 前景(>0)质心为中心裁一个固定子窗。
        output: (B,C,D,H,W), target_eval: (B,1,D,H,W)
        """
        if crop_size <= 0:
            return output, target_eval
        d, h, w = output.shape[2:]
        cd = min(crop_size, d)
        ch = min(crop_size, h)
        cw = min(crop_size, w)
        out_chunks = []
        tgt_chunks = []
        for bi in range(output.shape[0]):
            fg = (target_eval[bi, 0] > 0)
            if torch.any(fg):
                pts = torch.nonzero(fg, as_tuple=False).float()
                center = pts.mean(0)
            else:
                center = torch.tensor(
                    [(d - 1) / 2.0, (h - 1) / 2.0, (w - 1) / 2.0],
                    device=output.device,
                    dtype=torch.float32,
                )
            cdi = int(torch.round(center[0]).item())
            chi = int(torch.round(center[1]).item())
            cwi = int(torch.round(center[2]).item())
            d0 = min(max(0, cdi - cd // 2), d - cd)
            h0 = min(max(0, chi - ch // 2), h - ch)
            w0 = min(max(0, cwi - cw // 2), w - cw)
            d1, h1, w1 = d0 + cd, h0 + ch, w0 + cw
            out_chunks.append(output[bi : bi + 1, :, d0:d1, h0:h1, w0:w1])
            tgt_chunks.append(target_eval[bi : bi + 1, :, d0:d1, h0:h1, w0:w1])
        return torch.cat(out_chunks, dim=0), torch.cat(tgt_chunks, dim=0)

    def _compute_hard_stats(self, output, target_eval):
        axes = [0] + list(range(2, output.ndim))
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            # float32: 与 get_tp_fp_fn_tn 内 y_onehot 一致，避免 fp16 与 TP/FP 乘加数值问题
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target_eval != self.label_manager.ignore_label).float()
                target_eval = target_eval.clone()
                target_eval[target_eval == self.label_manager.ignore_label] = 0
            else:
                if target_eval.dtype == torch.bool:
                    mask = ~target_eval[:, -1:]
                else:
                    mask = 1 - target_eval[:, -1:]
                target_eval = target_eval[:, :-1]
        else:
            mask = None

        from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_eval, axes=axes, mask=mask)
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]
        return tp_hard, fp_hard, fn_hard

    def _compute_loc_metrics(self, pred_centroid_norm, gt_centroid_norm, wb_shapes_or_padded):
        """
        pred_centroid_norm / gt_centroid_norm 必须在同一坐标系 (调用方负责保证).
        wb_shapes_or_padded 用作 voxel 单位换算:
          - 若是 (B, 3) per-case shapes (老路径 phase1 dataloader), 走原行为;
          - 若是 torch.Size / tuple (新路径: build_whole_brain_batch + padded centroid),
            则按 padded shape 统一换算到 padded-voxel 距离.
        """
        if isinstance(wb_shapes_or_padded, torch.Tensor) and wb_shapes_or_padded.ndim == 2:
            scale = wb_shapes_or_padded - 1
        else:
            d, h, w = (int(wb_shapes_or_padded[0]),
                       int(wb_shapes_or_padded[1]),
                       int(wb_shapes_or_padded[2]))
            scale = torch.tensor(
                [max(d - 1, 1), max(h - 1, 1), max(w - 1, 1)],
                device=pred_centroid_norm.device,
                dtype=torch.float32,
            )
        pred_vox = pred_centroid_norm * scale
        gt_vox = gt_centroid_norm * scale
        return torch.norm(pred_vox - gt_vox, dim=1).mean()

    def _prepare_seg_target_for_loss(self, target, target_full):
        if self.enable_deep_supervision and isinstance(target, list):
            return target
        return target_full

    def _init_phase2_vessel_keys(self):
        if not self.phase2_precomputed_dir or not self._case_mapping:
            return
        # 严格只在当前 fold 的 train keys 中建立过采样池，避免 train/val 泄漏
        train_key_whitelist = None
        try:
            if nnUNet_preprocessed is not None:
                split_file = os.path.join(
                    nnUNet_preprocessed,
                    self.plans_manager.dataset_name,
                    "splits_final.json",
                )
                if os.path.isfile(split_file) and isinstance(self.fold, int):
                    with open(split_file) as f:
                        splits = json.load(f)
                    if 0 <= int(self.fold) < len(splits):
                        train_key_whitelist = set(splits[int(self.fold)]["train"])
        except Exception:
            train_key_whitelist = None

        keys = []
        for nnunet_key, prepared_case in self._case_mapping.items():
            if train_key_whitelist is not None and nnunet_key not in train_key_whitelist:
                continue
            lbl_file = os.path.join(self.phase2_precomputed_dir, prepared_case, "label.npy")
            if not os.path.isfile(lbl_file):
                continue
            try:
                lbl = np.load(lbl_file, mmap_mode="r")
                vessel_vox = int((lbl == 2).sum())
            except Exception:
                continue
            if vessel_vox >= self.phase2_vessel_min_voxels:
                keys.append(nnunet_key)
        self._phase2_vessel_positive_keys = keys
        if self.phase2_precomputed_dir:
            self.print_to_log_file(
                f"[phase2-vessel] oversample_prob={self.phase2_vessel_oversample_prob}, "
                f"min_voxels={self.phase2_vessel_min_voxels}, positive_cases={len(keys)}, "
                f"train_only={train_key_whitelist is not None}"
            )

    def _maybe_oversample_phase2_keys(self, keys):
        if (
            self._get_phase_name() != "phase2"
            or not self.phase2_precomputed_dir
            or self.phase2_vessel_oversample_prob <= 0
            or len(self._phase2_vessel_positive_keys) == 0
        ):
            return keys
        out = []
        for k in keys:
            if random.random() < self.phase2_vessel_oversample_prob:
                out.append(random.choice(self._phase2_vessel_positive_keys))
            else:
                out.append(k)
        return out

    def _maybe_oversample_phase3_vessel_keys(self, keys):
        """与 phase2 逻辑相同, 用于 phase3 train; 复用 _phase2_vessel_positive_keys 清单。"""
        if (
            self.phase3_vessel_oversample_prob <= 0
            or len(self._phase2_vessel_positive_keys) == 0
        ):
            return keys
        out = []
        for k in keys:
            if random.random() < self.phase3_vessel_oversample_prob:
                out.append(random.choice(self._phase2_vessel_positive_keys))
            else:
                out.append(k)
        return out

    def _add_vessel_aux_loss(self, seg_output, seg_target):
        if self.vessel_loss_weight <= 0:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        logits = seg_output[0] if isinstance(seg_output, (tuple, list)) else seg_output
        target = seg_target[0] if isinstance(seg_target, list) else seg_target
        # target: (B,1,D,H,W) with classes {0,1,2}; vessel is class 2
        # one-vs-rest logit for class 2, AMP-safe with BCEWithLogits
        vessel_logit = logits[:, 2, ...] - torch.logsumexp(logits[:, [0, 1], ...], dim=1)
        vessel_gt = (target[:, 0, ...] == 2).float()
        bce = F.binary_cross_entropy_with_logits(vessel_logit, vessel_gt)
        return self.vessel_loss_weight * bce

    def _add_phase3_fg_aux_loss(self, seg_output, seg_target):
        """Phase3 前景 BCE 辅助: pos_weight 有上限(曾用过大值 + AMP 导致 smooth 等 run 的 seg→nan)。"""
        if self.phase3_fg_aux_weight <= 0:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        logits = seg_output[0] if isinstance(seg_output, (tuple, list)) else seg_output
        target = seg_target[0] if isinstance(seg_target, list) else seg_target
        if logits.shape[1] < 2:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        fg_logit = torch.logsumexp(logits[:, 1:, ...], dim=1) - logits[:, 0, ...]
        fg_logit = fg_logit.clamp(-30.0, 30.0)
        fg_gt = (target[:, 0, ...] > 0).float()
        n_pos = fg_gt.sum()
        if float(n_pos.item()) < 0.5:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        n_neg = fg_gt.numel() - n_pos
        pos_weight = (n_neg / n_pos).clamp(1.0, 80.0)
        bce = F.binary_cross_entropy_with_logits(
            fg_logit.float(), fg_gt.float(), pos_weight=pos_weight.float()
        )
        out = self.phase3_fg_aux_weight * bce
        if not torch.isfinite(out).all():
            return torch.zeros((), dtype=torch.float32, device=self.device)
        return out

    def _should_skip_seg_update(self, seg_loss: torch.Tensor):
        if not bool(torch.isfinite(seg_loss).all()):
            return True, "non-finite"
        if self.seg_loss_skip_threshold > 0:
            v = float(seg_loss.detach().item())
            if v > self.seg_loss_skip_threshold:
                return True, f"seg_loss>{self.seg_loss_skip_threshold}"
        return False, ""

    def _get_phase3_seg_transforms(self):
        """Lazily build nnU-Net-style augmentation pipeline for Phase3 seg patches."""
        if self._phase3_seg_transforms is not None:
            return self._phase3_seg_transforms

        from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
        from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
        from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
        from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
        from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
        from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
        from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
        from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
        from batchgeneratorsv2.transforms.utils.random import RandomTransform

        patch_size = self.configuration_manager.patch_size
        rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        mirror_axes = (0, 1, 2)

        transforms = []
        transforms.append(
            SpatialTransform(
                patch_size, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2, rotation=rotation_for_DA,
                p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False
            )
        )
        transforms.append(RandomTransform(
            GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True),
            apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(blur_sigma=(0.5, 1.), synchronize_channels=False,
                                  synchronize_axes=False, p_per_channel=0.5, benchmark=True),
            apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(multiplier_range=BGContrast((0.75, 1.25)),
                                             synchronize_channels=False, p_per_channel=1),
            apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(contrast_range=BGContrast((0.75, 1.25)), preserve_range=True,
                              synchronize_channels=False, p_per_channel=1),
            apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(scale=(0.5, 1), synchronize_channels=False,
                                          synchronize_axes=True, ignore_axes=None,
                                          allowed_channels=None, p_per_channel=0.5),
            apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(gamma=BGContrast((0.7, 1.5)), p_invert_image=1,
                           synchronize_channels=False, p_per_channel=1, p_retain_stats=1),
            apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(gamma=BGContrast((0.7, 1.5)), p_invert_image=0,
                           synchronize_channels=False, p_per_channel=1, p_retain_stats=1),
            apply_probability=0.3
        ))
        transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        self._phase3_seg_transforms = ComposeTransforms(transforms)
        self.print_to_log_file("[phase3-seg-aug] Built nnU-Net-style augmentation pipeline for Phase3 seg patches")
        return self._phase3_seg_transforms

    def _dump_train_batch_png(self, seg_input, seg_target_full, phase_keys):
        """DIAG: dump 一次 train batch 的 image+GT 三视图 PNG, 看网络实际见到的数据是否对齐."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out_dir = os.path.join(REPO_ROOT, 'diag_outputs')
        os.makedirs(out_dir, exist_ok=True)
        for bi in range(seg_input.shape[0]):
            img = seg_input[bi, 0].detach().cpu().numpy()
            gt = seg_target_full[bi, 0].detach().cpu().numpy()
            mask1 = (gt == 1)
            if mask1.sum() > 0:
                cz, cy, cx = np.argwhere(mask1).mean(axis=0).astype(int)
            else:
                cz = img.shape[0] // 2
                cy = img.shape[1] // 2
                cx = img.shape[2] // 2
            fig, ax = plt.subplots(3, 3, figsize=(15, 15))
            specs = [
                ("axial", img[cz], gt[cz]),
                ("coronal", img[:, cy, :], gt[:, cy, :]),
                ("sagittal", img[:, :, cx], gt[:, :, cx]),
            ]
            for r, (name, img2, gt2) in enumerate(specs):
                imn = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)
                ax[r, 0].imshow(imn, cmap='gray')
                ax[r, 0].set_title(f'{name} - image (after aug)')
                rgb = np.zeros((*gt2.shape, 3))
                rgb[gt2 == 1] = [1, 0, 0]
                rgb[gt2 == 2] = [0, 0, 1]
                ax[r, 1].imshow(rgb)
                ax[r, 1].set_title(f'{name} - GT')
                ov = np.stack([imn]*3, axis=-1)
                ov[gt2 == 1] = [1, 0.3, 0.3]
                ov[gt2 == 2] = [0.3, 0.3, 1]
                ax[r, 2].imshow(ov)
                ax[r, 2].set_title(f'{name} - overlay')
                py, px = np.array(img2.shape) // 2
                for c in range(3):
                    ax[r, c].axhline(py, color='lime', lw=0.5, alpha=0.5)
                    ax[r, c].axvline(px, color='lime', lw=0.5, alpha=0.5)
                    ax[r, c].axis('off')
            n_fg1 = int(mask1.sum())
            n_fg2 = int((gt == 2).sum())
            plt.suptitle(
                f'TRAIN batch[{bi}] case={phase_keys[bi]}\n'
                f'after augmentation: img mean={img.mean():.3f} std={img.std():.3f}\n'
                f'GT FG1={n_fg1}, FG2={n_fg2}, FG1 centroid in patch=({cz},{cy},{cx}) (lime cross = patch geom center)'
            )
            plt.tight_layout()
            out_path = f"{out_dir}/train_batch_iter0_b{bi}.png"
            plt.savefig(out_path, dpi=80, bbox_inches='tight')
            plt.close()
            self.print_to_log_file(f"[DIAG] dumped {out_path}  FG1={n_fg1}  FG2={n_fg2}")

    def _apply_phase3_seg_augmentation(self, seg_input, seg_target_full):
        """
        Apply nnU-Net augmentation to Phase3 seg patch (after crop, before forward_seg).
        seg_input: (B, C, D, H, W) float tensor on device
        seg_target_full: (B, 1, D, H, W) long tensor on device
        Returns augmented (seg_input, seg_target_full) on the same device.
        """
        transforms = self._get_phase3_seg_transforms()
        device = seg_input.device
        B = seg_input.shape[0]

        aug_data_list = []
        aug_seg_list = []
        for bi in range(B):
            img_tensor = seg_input[bi].cpu()          # (C, D, H, W)
            seg_tensor = seg_target_full[bi].cpu().to(torch.int16)  # (1, D, H, W)
            sample = {"image": img_tensor, "segmentation": seg_tensor}
            sample = transforms(**sample)
            aug_data_list.append(sample["image"].float())
            aug_seg_list.append(sample["segmentation"].long())

        seg_input_aug = torch.stack(aug_data_list, dim=0).to(device)
        seg_target_aug = torch.stack(aug_seg_list, dim=0).to(device)
        return seg_input_aug, seg_target_aug

    def _round_centroid_to_voxel(self, centroid_norm: torch.Tensor, padded_shape) -> torch.Tensor:
        """
        把 normalized centroid round 到最近的整数 voxel 位置后再 normalize 回 [0,1].

        修复 image 与 GT label 裁剪坐标错位的 bug:
          - image 裁剪 (differentiable_crop_3d, mode='bilinear') 用 float center 做 trilinear
          - GT label 裁剪 (_load_and_crop_gt_label_patch) 用 np.rint 强制 round 到 int
          → 同一个 centroid 输入, image 和 GT 之间最多差 0.5 voxel, 对 ~10 voxel 的 TN
            是系统性 30-50% 边界错位, phase3 dice 永远拉不上 phase2 的根因.

        通过提前把 centroid round 到整数 voxel, 让两边都用同一个整数中心,
        消除错位. 副作用: 切断 seg_loss 经 crop 反传到 loc head 的梯度
        (round 不可导), 但 loc 自身有 loc_loss, 不依赖此路径学习.

        通过 TN_PHASE3_INT_CENTROID=0 可关闭 (回到旧行为, 仅用于对比).
        """
        if os.environ.get("TN_PHASE3_INT_CENTROID", "1") not in ("1", "true", "True"):
            return centroid_norm
        if isinstance(padded_shape, torch.Tensor):
            shape_t = padded_shape.to(centroid_norm.device).float()
        else:
            shape_t = torch.tensor(
                [float(int(s)) for s in padded_shape],
                device=centroid_norm.device, dtype=torch.float32,
            )
        scale = torch.clamp(shape_t - 1.0, min=1.0)  # (3,)
        # round 到整数 voxel (与 _load_and_crop_gt_label_patch 用的 np.rint 完全一致)
        center_vox = torch.round(centroid_norm * scale)
        # clamp 到 [0, dim-1]
        center_vox = torch.minimum(
            torch.maximum(center_vox, torch.zeros_like(center_vox)),
            shape_t - 1.0,
        )
        return (center_vox / scale).detach()

    def _phase3_mix_centroid(self, pred_centroid_norm, gt_centroid_norm):
        phase3_epoch = self.current_epoch - self.phase1_epochs - self.phase2_epochs
        if self.phase3_warmup_epochs <= 0:
            alpha = 1.0
        else:
            alpha = min(1.0, float(phase3_epoch + 1) / float(self.phase3_warmup_epochs))
        # 始终保留一部分 GT 锚定，避免 phase3 长训练后 crop 漂移导致分割塌缩为全背景
        gt_mix = min(max(self.phase3_min_gt_mix, 0.0), 1.0)
        alpha = min(alpha, 1.0 - gt_mix)
        return (1.0 - alpha) * gt_centroid_norm + alpha * pred_centroid_norm

    def _ensure_phase1_loaders(self):
        if self._phase1_train_loader is not None and self._phase1_val_loader is not None:
            return

        train_ids, val_ids = get_train_val_split(
            self.prepared_data_dir,
            train_ratio=self.phase1_train_val_split,
            seed=self.phase1_split_seed,
            case_filter=self.phase1_case_filter,
        )
        train_ds = TNLocSegDataset(
            self.prepared_data_dir,
            case_ids=train_ids,
            phase=1,
            transform=LegacyLocAugment(
                affine_prob=self.phase1_affine_prob,
                max_rotate_deg=self.phase1_max_rotate_deg,
                scale_min=self.phase1_scale_min,
                scale_max=self.phase1_scale_max,
                intensity_shift=self.phase1_intensity_shift,
                intensity_scale=self.phase1_intensity_scale,
                blur_prob=self.phase1_blur_prob,
            ),
            cache_wb=False,
            lr_flip_prob=self.phase1_lr_flip_prob,
            mask_size=None,
        )
        val_ds = TNLocSegDataset(
            self.prepared_data_dir,
            case_ids=val_ids,
            phase=1,
            cache_wb=False,
            lr_flip_prob=0.0,
            mask_size=None,
        )

        self._phase1_train_loader = DataLoader(
            train_ds,
            batch_size=self.loc_batch_size_phase1,
            shuffle=True,
            num_workers=self.phase1_num_workers,
            pin_memory=self.phase1_pin_memory,
            collate_fn=locseg_collate_fn,
            drop_last=True,
        )
        self._phase1_val_loader = DataLoader(
            val_ds,
            batch_size=self.loc_batch_size_phase1,
            shuffle=False,
            num_workers=self.phase1_num_workers,
            pin_memory=self.phase1_pin_memory,
            collate_fn=locseg_collate_fn,
            drop_last=True,
        )
        cf_log = f" case_filter={self.phase1_case_filter}" if self.phase1_case_filter else ""
        self.print_to_log_file(
            "[phase1-old-loader]"
            f"{cf_log} "
            f"train_cases={len(train_ds)} val_cases={len(val_ds)} "
            f"bs={self.loc_batch_size_phase1} workers={self.phase1_num_workers} "
            f"lr_flip_prob={self.phase1_lr_flip_prob} "
            f"affine_prob={self.phase1_affine_prob} rot={self.phase1_max_rotate_deg} "
            f"scale=[{self.phase1_scale_min},{self.phase1_scale_max}] "
            f"intensity_shift={self.phase1_intensity_shift} intensity_scale={self.phase1_intensity_scale} "
            f"blur_prob={self.phase1_blur_prob}"
        )
        self.print_to_log_file(f"[phase2] use_loc_crop={self.phase2_use_loc_crop}")
        if self.phase2_precomputed_dir:
            self.print_to_log_file(f"[phase2] precomputed_dir={self.phase2_precomputed_dir}")

    def _phase1_old_loc_step(self, batch, train: bool):
        phase = self._get_phase_name()
        self._apply_freeze_for_phase(phase)
        mod = self.network.module if self.is_ddp else self.network

        wb_tensor = batch["whole_brain"].to(self.device, non_blocking=True)
        gt_centroid_norm = batch["centroid_norm"].to(self.device, non_blocking=True)
        wb_shapes = batch["wb_shape"].to(self.device, non_blocking=True)

        seg_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            hm_l, hm_r, c_l, c_r = self._loc_forward(wb_tensor, mod)
            pred_centroid = self._select_centroid_by_gt_side(c_l, c_r, gt_centroid_norm)
            hm_gt = generate_heatmap_target(gt_centroid_norm, hm_l.shape[2:], sigma=self.loc_sigma).to(self.device)
            hm_pred = torch.where((gt_centroid_norm[:, 0] < 0.5).view(-1, 1, 1, 1, 1), hm_l, hm_r)
            loc_loss = self.loc_loss_fn(hm_pred, hm_gt, pred_centroid, gt_centroid_norm)
            loc_err_vox = self._compute_loc_metrics(pred_centroid, gt_centroid_norm, wb_shapes)
            total_loss = self.loc_loss_weight * loc_loss

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            if self.grad_scaler is not None:
                self.grad_scaler.scale(total_loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()

        return {
            "loss": total_loss.detach().cpu().numpy(),
            "seg_loss": seg_loss.detach().cpu().numpy(),
            "loc_loss": loc_loss.detach().cpu().numpy(),
            "loc_err_vox": loc_err_vox.detach().cpu().numpy(),
            "phase": phase,
        }

    def _load_phase2_precomputed_case(self, nnunet_key: str):
        if nnunet_key in self._phase2_crop_cache:
            item = self._phase2_crop_cache.pop(nnunet_key)
            self._phase2_crop_cache[nnunet_key] = item
            return item

        prepared_case = self._case_mapping.get(nnunet_key, None)
        if prepared_case is None:
            raise KeyError(f"missing case mapping for {nnunet_key}")
        if not self.phase2_precomputed_dir:
            raise RuntimeError("phase2_precomputed_dir is empty")

        case_dir = os.path.join(self.phase2_precomputed_dir, prepared_case)
        img_file = os.path.join(case_dir, "image.npy")
        lbl_file = os.path.join(case_dir, "label.npy")
        if not os.path.isfile(img_file) or not os.path.isfile(lbl_file):
            raise FileNotFoundError(
                f"missing precomputed phase2 files for {prepared_case}: image={img_file}, label={lbl_file}"
            )
        img = np.load(img_file).astype(np.float32)
        lbl = np.load(lbl_file).astype(np.int64)
        item = (img, lbl)
        self._phase2_crop_cache[nnunet_key] = item
        if len(self._phase2_crop_cache) > self.case_cache_size:
            self._phase2_crop_cache.popitem(last=False)
        return item

    def _build_phase2_precomputed_batch(self, keys):
        keys = self._maybe_oversample_phase2_keys(keys)
        imgs, lbls = [], []
        for k in keys:
            img, lbl = self._load_phase2_precomputed_case(k)
            imgs.append(img)
            lbls.append(lbl)
        img_np = np.stack(imgs, axis=0)[:, None]  # (B,1,D,H,W)
        lbl_np = np.stack(lbls, axis=0)[:, None]  # (B,1,D,H,W)
        # 数据兜底: 标签类别必须在 [0, 2]，避免 Dice one-hot/scatter CUDA 越界
        if lbl_np.max() > 2 or lbl_np.min() < 0:
            lbl_np = np.clip(lbl_np, 0, 2)
        img_t = torch.from_numpy(img_np).to(self.device, non_blocking=True)
        lbl_t = torch.from_numpy(lbl_np).to(self.device, non_blocking=True).long()
        return img_t, lbl_t

    @staticmethod
    def _build_seg_target_like_output(seg_output, target_full):
        if isinstance(seg_output, (tuple, list)):
            target_list = []
            for out in seg_output:
                if tuple(out.shape[2:]) == tuple(target_full.shape[2:]):
                    target_list.append(target_full)
                else:
                    down = F.interpolate(target_full.float(), size=out.shape[2:], mode="nearest").long()
                    target_list.append(down)
            return target_list
        return target_full

    @staticmethod
    def _slice_seg_target(seg_target, start: int, end: int):
        if isinstance(seg_target, list):
            return [t[start:end] for t in seg_target]
        return seg_target[start:end]

    def _phase2_val_pred_crop_chunked(self, keys, mod, patch_size, seg_target):
        """
        phase2 验证时，按 small chunks 走 whole-brain->loc(pred)->crop，避免一次性 OOM。
        """
        bsz = len(keys)
        chunk = max(1, int(self.phase2_val_wb_chunk_size))
        seg_out_chunks = []
        seg_loss_acc = torch.zeros((), dtype=torch.float32, device=self.device)
        loc_err_acc = torch.zeros((), dtype=torch.float32, device=self.device)

        for s in range(0, bsz, chunk):
            e = min(bsz, s + chunk)
            keys_chunk = keys[s:e]
            wb_tensor, wb_shapes, gt_centroid_norm = self._build_whole_brain_batch(keys_chunk)
            padded_shape = wb_tensor.shape[2:]
            hm_l, hm_r, c_l, c_r = self._loc_forward(wb_tensor, mod)
            pred_centroid = self._select_centroid_by_gt_side(
                c_l, c_r, gt_centroid_norm, wb_shapes=wb_shapes, padded_shape=padded_shape
            )
            loc_err_chunk = self._compute_loc_metrics(pred_centroid, gt_centroid_norm, padded_shape)
            seg_input = self._crop_with_centroid(wb_tensor, pred_centroid.detach(), patch_size)
            seg_out_chunk = mod.forward_seg(seg_input)
            target_chunk = self._slice_seg_target(seg_target, s, e)
            seg_loss_chunk = self.loss(seg_out_chunk, target_chunk)

            w = float(e - s) / float(max(1, bsz))
            seg_loss_acc = seg_loss_acc + seg_loss_chunk * w
            loc_err_acc = loc_err_acc + loc_err_chunk * w
            seg_out_chunks.append(seg_out_chunk)

            # 尽早释放 whole-brain chunk 显存，降低峰值
            del wb_tensor, wb_shapes, gt_centroid_norm, hm_l, hm_r, c_l, c_r, pred_centroid, seg_input

        if isinstance(seg_out_chunks[0], (tuple, list)):
            n_levels = len(seg_out_chunks[0])
            seg_output = [torch.cat([c[i] for c in seg_out_chunks], dim=0) for i in range(n_levels)]
        else:
            seg_output = torch.cat(seg_out_chunks, dim=0)
        return seg_output, seg_loss_acc, loc_err_acc

    def _shared_step(self, batch, train: bool):
        data = batch["data"].to(self.device, non_blocking=True)
        target_raw = batch["target"]
        keys = batch.get("keys", None)
        if keys is None:
            raise KeyError("nnU-Net batch is missing `keys`, cannot bridge whole-brain data")

        if isinstance(target_raw, list):
            target = [i.to(self.device, non_blocking=True) for i in target_raw]
            target_full = target[0]
        else:
            target = target_raw.to(self.device, non_blocking=True)
            target_full = target

        patch_size = tuple(int(i) for i in target_full.shape[2:])
        phase = self._get_phase_name()
        self._apply_freeze_for_phase(phase)

        mod = self.network.module if self.is_ddp else self.network

        seg_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        loc_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        loc_err_vox = torch.zeros((), dtype=torch.float32, device=self.device)
        seg_output = None

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            if phase == "phase2":
                if self.phase2_precomputed_dir:
                    if (not train) and self.phase2_val_use_pred_crop:
                        seg_target = self._prepare_seg_target_for_loss(target, target_full)
                        seg_output, seg_loss, loc_err_vox = self._phase2_val_pred_crop_chunked(
                            keys, mod, patch_size, seg_target
                        )
                        target = seg_target
                    else:
                        # 训练可使用预处理 crop（例如加速/稳定）
                        seg_input, target_full = self._build_phase2_precomputed_batch(keys)
                        seg_output = mod.forward_seg(seg_input)
                        target = self._build_seg_target_like_output(seg_output, target_full)
                elif self.phase2_use_loc_crop:
                    # phase2: 使用 phase1 定位网络预测中心后裁剪 ROI 进行分割训练
                    wb_tensor, wb_shapes, gt_centroid_norm = self._build_whole_brain_batch(keys)
                    padded_shape = wb_tensor.shape[2:]
                    with torch.no_grad():
                        hm_l, hm_r, c_l, c_r = self._loc_forward(wb_tensor, mod)
                        pred_centroid = self._select_centroid_by_gt_side(
                            c_l, c_r, gt_centroid_norm, wb_shapes=wb_shapes, padded_shape=padded_shape
                        )
                    loc_err_vox = self._compute_loc_metrics(pred_centroid, gt_centroid_norm, padded_shape)
                    seg_input = self._crop_with_centroid(wb_tensor, pred_centroid.detach(), patch_size)
                    seg_output = mod.forward_seg(seg_input)
                    seg_target = self._prepare_seg_target_for_loss(target, target_full)
                    target = seg_target
                else:
                    # 回退路径: phase2 纯 nnU-Net patch 训练
                    seg_output = mod.forward_seg(data)
                    seg_target = self._prepare_seg_target_for_loss(target, target_full)
                    target = seg_target
                if not ((phase == "phase2") and (self.phase2_precomputed_dir) and ((not train) and self.phase2_val_use_pred_crop)):
                    seg_loss = self.loss(seg_output, target)
                if phase == "phase2" and train:
                    seg_loss = seg_loss + self._add_vessel_aux_loss(seg_output, target)
                total_loss = seg_loss
            else:
                phase_keys = keys
                if phase == "phase1":
                    # phase1 先抽样 keys 再读 whole-brain，避免把完整 nnU-Net batch 全搬到 GPU
                    phase_keys = self._subsample_keys_for_phase1_loc(keys)
                elif phase == "phase3" and train:
                    phase_keys = self._maybe_oversample_phase3_vessel_keys(keys)

                wb_tensor, wb_shapes, gt_centroid_norm = self._build_whole_brain_batch(phase_keys)
                padded_shape = wb_tensor.shape[2:]
                # 给 loc 输入加 augmentation (rotation+scale around centroid + intensity)
                # 默认关 (TN_LOC_PHASE3_AUG=0). 小数据 fine-tune 强烈推荐开.
                # centroid 是 affine fixed point, 不变. 只影响 loc forward, seg crop 仍用原 wb.
                if train and phase == "phase3" and self.loc_phase3_aug_enabled:
                    wb_for_loc = self._apply_loc_phase3_augmentation(wb_tensor, gt_centroid_norm)
                else:
                    wb_for_loc = wb_tensor
                hm_l, hm_r, c_l, c_r = self._loc_forward(wb_for_loc, mod)
                pred_centroid = self._select_centroid_by_gt_side(
                    c_l, c_r, gt_centroid_norm, wb_shapes=wb_shapes, padded_shape=padded_shape
                )
                hm_gt = generate_heatmap_target(gt_centroid_norm, hm_l.shape[2:], sigma=self.loc_sigma)
                hm_gt = hm_gt.to(self.device)
                left_mask = self._gt_left_mask_padded(gt_centroid_norm, wb_shapes, padded_shape)
                hm_pred = torch.where(left_mask.view(-1, 1, 1, 1, 1), hm_l, hm_r)
                loc_loss = self.loc_loss_fn(hm_pred, hm_gt, pred_centroid, gt_centroid_norm)
                loc_err_vox = self._compute_loc_metrics(pred_centroid, gt_centroid_norm, padded_shape)

                if phase == "phase1":
                    total_loss = self.loc_loss_weight * loc_loss
                else:
                    if (not train) and self.phase3_val_use_gt_crop:
                        centroid_for_crop = gt_centroid_norm
                    else:
                        centroid_for_crop = self._phase3_mix_centroid(pred_centroid, gt_centroid_norm)
                    # FIX: 把 centroid round 到整数 voxel, 让 image (bilinear) 与 GT (np.rint)
                    # 裁在同一个位置, 消除 sub-voxel 错位 bug.
                    # 关闭: export TN_PHASE3_INT_CENTROID=0
                    centroid_for_crop = self._round_centroid_to_voxel(
                        centroid_for_crop, wb_tensor.shape[2:]
                    )
                    seg_input = self._crop_with_centroid(wb_tensor, centroid_for_crop, patch_size)
                    seg_target_full = self._load_and_crop_gt_label_patch(
                        phase_keys,
                        wb_shapes,
                        wb_tensor.shape[2:],
                        centroid_for_crop,
                        patch_size,
                    )
                    if train and self.phase3_seg_aug_enabled:
                        seg_input, seg_target_full = self._apply_phase3_seg_augmentation(
                            seg_input.detach().clone(), seg_target_full
                        )
                    # === DIAG (一次性): TN_DUMP_BATCH=1 时 dump 第 1 个 iter 看实际数据 ===
                    if train and os.environ.get("TN_DUMP_BATCH", "0") in ("1", "true", "True"):
                        if not hasattr(self, "_diag_dumped"):
                            self._diag_dumped = True
                            try:
                                self._dump_train_batch_png(seg_input, seg_target_full, phase_keys)
                            except Exception as _e:
                                self.print_to_log_file(f"[DIAG] dump failed: {_e}")
                    seg_output = mod.forward_seg(seg_input)
                    seg_target = self._build_seg_target_like_output(seg_output, seg_target_full)
                    # 修 bug (#26): pseudo dice 计算用 result["tp_hard"]/[fp_hard]/[fn_hard],
                    # 它在下方读 `target` 而不是 `seg_target`. phase2 路径都做了 target=seg_target
                    # 重赋值, 但 phase3 漏了这一行 → pseudo dice 跟 dataloader 给的 target (随机
                    # 裁的 patch, 不是我们 wholebrain GT 裁的) 比, 永远低. 修复: 让 target 跟随
                    # seg_target, 与 phase2 路径行为一致.
                    target = seg_target
                    seg_loss = self.loss(seg_output, seg_target)
                    if train:
                        seg_loss = seg_loss + self._add_phase3_fg_aux_loss(seg_output, seg_target)
                        seg_loss = seg_loss + self._add_vessel_aux_loss(seg_output, seg_target)
                    total_loss = seg_loss + self.loc_loss_weight * loc_loss

        # seg 非有限时仅用 loc 反传 (原逻辑整步 skip 会导致 phase3 长时间不更新, 如 smooth 日志)
        phase3_seg_broken = bool(
            train
            and phase == "phase3"
            and seg_output is not None
            and (not torch.isfinite(seg_loss).all())
        )
        if phase3_seg_broken:
            total_loss = self.loc_loss_weight * loc_loss

        should_update = True
        skip_reason = ""
        if train and phase in ("phase2", "phase3") and self.skip_bad_seg_loss:
            if phase3_seg_broken:
                should_update = True
            else:
                should_update, skip_reason = self._should_skip_seg_update(seg_loss)
                should_update = not should_update
                if not should_update:
                    self._skip_bad_seg_loss_count_epoch += 1
                    if self._skip_bad_seg_loss_count_epoch <= 5:
                        self.print_to_log_file(
                            f"[skip-step] epoch={self.current_epoch} phase={phase} reason={skip_reason}"
                        )

        if train and should_update:
            self.optimizer.zero_grad(set_to_none=True)
            if self.grad_scaler is not None:
                self.grad_scaler.scale(total_loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12.0)
                self.optimizer.step()

        result = {
            "loss": total_loss.detach().cpu().numpy(),
            "seg_loss": seg_loss.detach().cpu().numpy(),
            "loc_loss": loc_loss.detach().cpu().numpy(),
            "loc_err_vox": loc_err_vox.detach().cpu().numpy(),
            "phase": phase,
            "skipped_update": np.int64(0 if should_update else 1),
        }

        if seg_output is not None:
            output = seg_output
            target_eval = target
            if self.enable_deep_supervision and isinstance(output, (tuple, list)):
                output = output[0]
                target_eval = target[0] if isinstance(target, list) else target
            elif isinstance(target, list):
                target_eval = target[0]

            tp_hard, fp_hard, fn_hard = self._compute_hard_stats(output, target_eval)
            result["tp_hard"] = tp_hard
            result["fp_hard"] = fp_hard
            result["fn_hard"] = fn_hard
            # 诊断 Dice=0: 统计预测前景体素、GT 前景体素及其交集
            pred_seg = output.argmax(1)
            gt_seg = target_eval[:, 0].long()
            pred_fg = (pred_seg > 0)
            gt_fg = (gt_seg > 0)
            inter_fg = pred_fg & gt_fg
            result["pred_fg_vox"] = np.int64(pred_fg.sum().detach().cpu().item())
            result["gt_fg_vox"] = np.int64(gt_fg.sum().detach().cpu().item())
            result["inter_fg_vox"] = np.int64(inter_fg.sum().detach().cpu().item())
            if self.inner_dice_enable and (not self.label_manager.has_regions):
                out_inner, tgt_inner = self._crop_around_target_foreground_centroid(
                    output, target_eval, self.inner_dice_crop_size
                )
                tp_i, fp_i, fn_i = self._compute_hard_stats(out_inner, tgt_inner)
                result["tp_hard_inner"] = tp_i
                result["fp_hard_inner"] = fp_i
                result["fn_hard_inner"] = fn_i

        return result

    def train_step(self, batch: dict) -> dict:
        return self._shared_step(batch, train=True)

    def validation_step(self, batch: dict) -> dict:
        with torch.no_grad():
            return self._shared_step(batch, train=False)

    def _train_iters_for_phase(self, phase: str) -> int:
        if phase == "phase3":
            return self.num_iterations_per_epoch_phase3
        return self.num_iterations_per_epoch

    def _val_iters_for_phase(self, phase: str) -> int:
        if phase == "phase3":
            base = self.num_val_iterations_per_epoch_phase3
            if not self.phase3_val_full_coverage:
                return base
            # full-coverage: ceil(n_val_cases * cover / batch_size), 至少 base
            if self._phase3_val_iters_cached is None:
                try:
                    _, val_keys = self.do_split()
                    n_val = len(val_keys)
                except Exception as exc:
                    self.print_to_log_file(
                        f"[val-full-coverage] do_split() failed ({type(exc).__name__}): "
                        f"fallback to base={base}"
                    )
                    self._phase3_val_iters_cached = base
                    return base
                bs = max(1, int(getattr(self, "batch_size", 1) or 1))
                needed = (n_val * self.phase3_val_coverage_factor + bs - 1) // bs
                full = max(base, needed)
                self._phase3_val_iters_cached = full
                self.print_to_log_file(
                    f"[val-full-coverage] n_val_cases={n_val} bs={bs} "
                    f"cover={self.phase3_val_coverage_factor} -> val_iters={full} "
                    f"(was {base})"
                )
            return self._phase3_val_iters_cached
        return self.num_val_iterations_per_epoch

    def on_epoch_start(self):
        super().on_epoch_start()
        phase = self._get_phase_name()
        if phase != self._last_logged_phase:
            self.print_to_log_file(f"[phase-scheduler] switch to {phase} @ epoch={self.current_epoch}")
            self.print_to_log_file(
                f"[loc-sigma] loc_input_size={self.loc_input_size}, "
                f"effective_sigma={self.loc_sigma:.3f}"
            )
            if self.smooth_sigma > 0:
                self.print_to_log_file(f"[smooth] Gaussian sigma={self.smooth_sigma} voxel")
            if phase == "phase3":
                self.print_to_log_file(
                    f"[phase3-comparable] min_gt_mix={self.phase3_min_gt_mix}, loc_loss_w={self.loc_loss_weight}, "
                    f"vessel_bce_w={self.vessel_loss_weight}, vessel_oversample_p={self.phase3_vessel_oversample_prob}, "
                    f"val_gt_crop={self.phase3_val_use_gt_crop}, "
                    f"fg_oversample_pct={self.oversample_foreground_percent}"
                )
            self._last_logged_phase = phase

    def on_train_epoch_start(self):
        """
        混合策略:
        - phase1: loc 使用独立 poly 学习率, seg lr=0
        - phase2: seg 使用 nnU-Net scheduler 学习率, loc lr=0
        - phase3: seg 继续 nnU-Net 学习率, loc 使用较小固定学习率
        """
        phase = self._get_phase_name()
        self._skip_bad_seg_loss_count_epoch = 0
        self.network.train()
        if self._optimizer_built_for_phase != phase:
            self._rebuild_optimizer_for_phase(phase)
        self.lr_scheduler.step(self.current_epoch)

        seg_sched_lr = 0.0
        for pg in self.optimizer.param_groups:
            if pg.get("name", "") == "seg":
                seg_sched_lr = pg["lr"]
                break
        if phase == "phase1":
            p1_progress = self.current_epoch / max(1, self.phase1_epochs)
            if self.loc_phase1_lr_schedule == "cosine":
                lr_min = self.loc_phase1_lr * self.loc_phase1_lr_min_ratio
                loc_lr = lr_min + 0.5 * (self.loc_phase1_lr - lr_min) * (1.0 + math.cos(math.pi * p1_progress))
            else:
                loc_lr = self.loc_phase1_lr * ((1.0 - p1_progress) ** 0.9)
            seg_lr = 0.0
            loc_wd = self.loc_phase1_weight_decay
        elif phase == "phase2":
            seg_lr = seg_sched_lr
            loc_lr = 0.0
            loc_wd = self.weight_decay
        else:
            seg_lr = seg_sched_lr
            phase3_epoch = max(0, self.current_epoch - self.phase1_epochs - self.phase2_epochs)
            p3_progress = phase3_epoch / max(1, self.phase3_epochs)
            if self.loc_phase3_lr_schedule == "cosine":
                lr_min = self.loc_phase3_lr * self.loc_phase3_lr_min_ratio
                loc_lr = lr_min + 0.5 * (self.loc_phase3_lr - lr_min) * (1.0 + math.cos(math.pi * p3_progress))
            elif self.loc_phase3_lr_schedule == "poly":
                loc_lr = self.loc_phase3_lr * ((1.0 - p3_progress) ** 0.9)
            else:
                loc_lr = self.loc_phase3_lr
            loc_wd = self.weight_decay

        for pg in self.optimizer.param_groups:
            name = pg.get("name", "")
            if name == "seg":
                pg["lr"] = seg_lr
                pg["weight_decay"] = self.weight_decay
            elif name == "loc":
                pg["lr"] = loc_lr
                pg["weight_decay"] = loc_wd

        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(
            f"Current learning rate: seg={np.round(seg_lr, 6)} | loc={np.round(loc_lr, 6)} "
            f"| loc_wd={loc_wd:.1e}"
        )
        self.logger.log("lrs", seg_lr, self.current_epoch)

    def _reduce_numpy_metric(self, arr):
        if self.is_ddp:
            g = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(g, arr)
            return np.vstack(g).mean()
        return np.mean(arr)

    def _sum_numpy_vector(self, arr):
        vec = np.sum(arr, axis=0)
        if self.is_ddp:
            g = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(g, vec)
            vec = np.vstack(g).sum(0)
        return vec

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        outputs = collate_outputs(train_outputs)
        train_seg_loss = self._reduce_numpy_metric(outputs["seg_loss"])
        train_loc_loss = self._reduce_numpy_metric(outputs["loc_loss"])
        train_loc_err_vox = self._reduce_numpy_metric(outputs["loc_err_vox"])
        self.print_to_log_file("train_seg_loss", np.round(train_seg_loss, 4))
        self.print_to_log_file("train_loc_loss", np.round(train_loc_loss, 4))
        self.print_to_log_file("train_loc_err_vox", np.round(train_loc_err_vox, 4))
        if "skipped_update" in outputs:
            skipped = int(np.sum(outputs["skipped_update"]))
            self.print_to_log_file(f"train_skipped_updates {skipped}")

    def _save_phase_best_if_needed(self, phase: str, val_seg_loss: float, val_loc_err_vox: float):
        if phase == "phase1":
            metric = -float(val_loc_err_vox)
        elif phase == "phase2":
            metric = -float(val_seg_loss)
        else:
            metric = -(float(val_seg_loss) + 0.1 * float(val_loc_err_vox))

        old = self._best_phase_metric[phase]
        if old is None or metric > old:
            self._best_phase_metric[phase] = metric
            name = os.path.join(self.output_folder, f"{phase}_best.pth")
            self.print_to_log_file(f"[logging-checkpoint] new {phase} best, saving: {name}")
            self.save_checkpoint(name)

    def on_validation_epoch_end(self, val_outputs):
        outputs = collate_outputs(val_outputs)
        phase = self._get_phase_name()

        # phase1 为 loc-only 验证, val 输出不包含 tp_hard/fp_hard/fn_hard
        if phase != "phase1":
            super().on_validation_epoch_end(val_outputs)
        else:
            val_loss_here = self._reduce_numpy_metric(outputs["loss"])
            self.logger.log("val_losses", val_loss_here, self.current_epoch)
            self.logger.log("mean_fg_dice", np.nan, self.current_epoch)
            self.logger.log("dice_per_class_or_region", [np.nan], self.current_epoch)

        val_seg_loss = self._reduce_numpy_metric(outputs["seg_loss"])
        val_loc_loss = self._reduce_numpy_metric(outputs["loc_loss"])
        val_loc_err_vox = self._reduce_numpy_metric(outputs["loc_err_vox"])
        self.print_to_log_file("val_seg_loss", np.round(val_seg_loss, 4))
        self.print_to_log_file("val_loc_loss", np.round(val_loc_loss, 4))
        self.print_to_log_file("val_loc_err_vox", np.round(val_loc_err_vox, 4))
        if "pred_fg_vox" in outputs:
            pred_fg = float(np.mean(outputs["pred_fg_vox"]))
            gt_fg = float(np.mean(outputs["gt_fg_vox"]))
            inter_fg = float(np.mean(outputs["inter_fg_vox"]))
            self.print_to_log_file(
                f"val_fg_stats pred_fg_vox={pred_fg:.1f} gt_fg_vox={gt_fg:.1f} inter_fg_vox={inter_fg:.1f}"
            )
        if "tp_hard_inner" in outputs:
            tp_i = self._sum_numpy_vector(outputs["tp_hard_inner"])
            fp_i = self._sum_numpy_vector(outputs["fp_hard_inner"])
            fn_i = self._sum_numpy_vector(outputs["fn_hard_inner"])
            dice_i = (2.0 * tp_i) / np.clip((2.0 * tp_i + fp_i + fn_i), 1e-8, None)
            self.print_to_log_file(
                f"val_inner{self.inner_dice_crop_size}_dice {np.round(dice_i, 4).tolist()}"
            )
            self.print_to_log_file(
                f"val_inner{self.inner_dice_crop_size}_mean_fg_dice {np.round(float(np.nanmean(dice_i)), 4)}"
            )
        self._save_phase_best_if_needed(self._get_phase_name(), val_seg_loss, val_loc_err_vox)

    def _log_validation_placeholder(self):
        """
        nnUNet logger 要求每个 epoch 的 val key 都有一条记录。
        非验证 epoch 复制上一条，保持日志长度一致，避免断言错误。
        """
        if len(self.logger.my_fantastic_logging["val_losses"]) > 0:
            prev_val = self.logger.my_fantastic_logging["val_losses"][-1]
            prev_mean = self.logger.my_fantastic_logging["mean_fg_dice"][-1]
            prev_dice = self.logger.my_fantastic_logging["dice_per_class_or_region"][-1]
        else:
            prev_val = np.nan
            prev_mean = np.nan
            prev_dice = [np.nan]
        self.logger.log("val_losses", prev_val, self.current_epoch)
        self.logger.log("mean_fg_dice", prev_mean, self.current_epoch)
        self.logger.log("dice_per_class_or_region", prev_dice, self.current_epoch)
        self.print_to_log_file(
            f"[val-schedule] skip validation @ epoch={self.current_epoch}, interval={self.val_interval}"
        )

    def run_training(self):
        self.on_train_start()
        show_pbar = self.local_rank == 0

        for _ in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            phase = self._get_phase_name()
            if phase == "phase1":
                self._ensure_phase1_loaders()

            self.on_train_epoch_start()
            train_outputs = []
            if phase == "phase1":
                train_iter = tqdm(
                    self._phase1_train_loader,
                    desc=f"Train {phase} e{self.current_epoch}",
                    leave=False,
                    dynamic_ncols=True,
                    disable=not show_pbar,
                )
                for batch in train_iter:
                    train_outputs.append(self._phase1_old_loc_step(batch, train=True))
            else:
                n_train_iters = self._train_iters_for_phase(phase)
                train_iter = trange(
                    n_train_iters,
                    desc=f"Train {phase} e{self.current_epoch}",
                    leave=False,
                    dynamic_ncols=True,
                    disable=not show_pbar,
                )
                for _ in train_iter:
                    train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            if phase == "phase1":
                # 旧 phase1 逻辑: 每 5 个 epoch 验证一次, 且在 phase1 末尾强制验证
                do_val = (((self.current_epoch + 1) % 5) == 0) or (
                    self.current_epoch == (self.phase1_epochs - 1)
                )
            else:
                do_val = ((self.current_epoch + 1) % max(1, self.val_interval) == 0) or (
                    self.current_epoch == (self.num_epochs - 1)
                )
            if do_val:
                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = []
                    if phase == "phase1":
                        val_iter = tqdm(
                            self._phase1_val_loader,
                            desc=f"Val   {phase} e{self.current_epoch}",
                            leave=False,
                            dynamic_ncols=True,
                            disable=not show_pbar,
                        )
                        for batch in val_iter:
                            val_outputs.append(self._phase1_old_loc_step(batch, train=False))
                    else:
                        n_val_iters = self._val_iters_for_phase(phase)
                        val_iter = trange(
                            n_val_iters,
                            desc=f"Val   {phase} e{self.current_epoch}",
                            leave=False,
                            dynamic_ncols=True,
                            disable=not show_pbar,
                        )
                        for _ in val_iter:
                            val_outputs.append(self.validation_step(next(self.dataloader_val)))
                    self.on_validation_epoch_end(val_outputs)
            else:
                self._log_validation_placeholder()

            self.on_epoch_end()

        self.on_train_end()
