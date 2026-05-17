"""
diag_minimal_overfit.py
=======================
最简可控的过拟合实验:
  - 不用 trainer / 不用 dataloader / 不用 augmentation / 不用 loc / 不用 deep supervision
  - 直接拿我们 diag_visual_alignment 验证过对齐的 image patch + GT patch
  - 用 PlainConvUNet 简化版 (与 D096 同款) 跑 200 个 SGD step
  - 看 train dice 能不能上去

如果这个最简流程都跑不上去 → 数据问题
如果这个跑得上去, 但 trainer 的 sanity 跑不上去 → trainer 复杂性出 bug
"""
import os
REPO_ROOT = os.environ.get('TN_ROOT', os.path.dirname(os.path.abspath(__file__)))
import json, os, sys
sys.path.insert(0, REPO_ROOT)

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.differentiable_crop import differentiable_crop_3d
from dynamic_network_architectures.architectures.unet import PlainConvUNet

device = torch.device('cuda:0')
torch.manual_seed(42); np.random.seed(42)

PREP = os.path.join(REPO_ROOT, 'nnUNet_data/nnUNet_preprocessed/Dataset001_TN')
PREPARED = os.path.join(REPO_ROOT, 'prepared_data')
case_mapping = json.load(open(os.path.join(REPO_ROOT, 'nnUNet_data/nnUNet_raw/Dataset001_TN/case_mapping.json')))

# === 复用 diag_visual_alignment 的逻辑 prepare image+GT patch ===
NNUNET_KEY = 'TN_0004'
prep_name = case_mapping[NNUNET_KEY]
info = json.load(open(f'{PREPARED}/{prep_name}/info.json'))
wb = nib.load(info['nii_path']).get_fdata().astype(np.float32)
if wb.ndim == 4: wb = wb[..., 0]
spacing_prep = np.array(nib.load(info['nii_path']).header.get_zooms()[:3], dtype=np.float64)

target_spacing = np.array([1.0, 1.0, 1.0])
out_shape = tuple(int(x) for x in np.maximum(1, np.round(np.array(wb.shape) * spacing_prep / target_spacing)))
wb_t = torch.from_numpy(wb).view(1, 1, *wb.shape)
wb_t = F.interpolate(wb_t, size=out_shape, mode='trilinear', align_corners=False)
wb = wb_t[0, 0].numpy()
mean = float(wb.mean()); std = float(wb.std())
wb = (wb - mean) / (std + 1e-8)

centroid_native = np.load(f'{PREPARED}/{prep_name}/centroid.npy').astype(np.float64)
gt_centroid_resampled = centroid_native * (spacing_prep / target_spacing)
wb_shape_arr = np.array(wb.shape, dtype=np.float32)
gt_centroid_norm = gt_centroid_resampled / np.maximum(wb_shape_arr - 1.0, 1.0)

PATCH = 96
wb_tensor = torch.from_numpy(wb).view(1, 1, *wb.shape)
center_norm_t = torch.from_numpy(gt_centroid_norm.astype(np.float32)).view(1, 3)
scale = torch.tensor([max(s - 1, 1) for s in wb.shape], dtype=torch.float32)
center_vox_t = torch.round(center_norm_t * scale).clamp(torch.zeros(3), scale)
center_norm_rounded = center_vox_t / scale

img_patch = differentiable_crop_3d(
    wb_tensor, center_norm_rounded, PATCH, mode='bilinear', padding_mode='border'
)[0, 0].numpy()

# GT patch (same as #25 fix)
seg_nii = nib.load(f'{PREP}/gt_segmentations/{NNUNET_KEY}.nii.gz')
seg = np.rint(seg_nii.get_fdata()).astype(np.int64)
if seg.ndim == 4: seg = seg[..., 0]
spacing_nnu = np.array(seg_nii.header.get_zooms()[:3], dtype=np.float64)
target_shape = np.maximum(1, np.round(np.array(seg.shape) * spacing_nnu / spacing_prep).astype(int))
seg_t = torch.from_numpy(seg.astype(np.float32)).view(1, 1, *seg.shape)
seg_t = F.interpolate(seg_t, size=tuple(int(x) for x in target_shape), mode='nearest')
seg_resamp = torch.round(seg_t[0, 0]).long().numpy()

native_shape = tuple(int(x) for x in nib.load(info['nii_path']).shape[:3])
center_native = np.round(centroid_native).astype(np.int64)
half_n = target_shape // 2
full_native = np.zeros(native_shape, dtype=np.int64)
def emb(c, h, t, dim):
    lo_full = c - h; hi_full = lo_full + t
    lo_clip = max(0, int(lo_full)); hi_clip = min(int(dim), int(hi_full))
    if hi_clip <= lo_clip: return None, None
    src_lo = lo_clip - int(lo_full); src_hi = src_lo + (hi_clip - lo_clip)
    return slice(lo_clip, hi_clip), slice(src_lo, src_hi)
s0, ss0 = emb(center_native[0], half_n[0], target_shape[0], native_shape[0])
s1, ss1 = emb(center_native[1], half_n[1], target_shape[1], native_shape[1])
s2, ss2 = emb(center_native[2], half_n[2], target_shape[2], native_shape[2])
full_native[s0, s1, s2] = seg_resamp[ss0, ss1, ss2]

if native_shape != out_shape:
    full_t = torch.from_numpy(full_native.astype(np.float32)).view(1, 1, *native_shape)
    full_t = F.interpolate(full_t, size=out_shape, mode='nearest')
    seg_full = torch.round(full_t[0, 0]).long().numpy()
else:
    seg_full = full_native

center_vox_int = center_vox_t.long()[0].numpy()
halves = [PATCH // 2] * 3
crops = [PATCH] * 3
case_dims = wb.shape
lo_raw = [int(center_vox_int[d]) - halves[d] for d in range(3)]
hi_raw = [lo_raw[d] + crops[d] for d in range(3)]
gt_patch = np.zeros((PATCH, PATCH, PATCH), dtype=np.int64)
d_lo = max(0, lo_raw[0]); d_hi = min(case_dims[0], hi_raw[0])
h_lo = max(0, lo_raw[1]); h_hi = min(case_dims[1], hi_raw[1])
w_lo = max(0, lo_raw[2]); w_hi = min(case_dims[2], hi_raw[2])
pd_lo = d_lo - lo_raw[0]; pd_hi = pd_lo + (d_hi - d_lo)
ph_lo = h_lo - lo_raw[1]; ph_hi = ph_lo + (h_hi - h_lo)
pw_lo = w_lo - lo_raw[2]; pw_hi = pw_lo + (w_hi - w_lo)
gt_patch[pd_lo:pd_hi, ph_lo:ph_hi, pw_lo:pw_hi] = seg_full[d_lo:d_hi, h_lo:h_hi, w_lo:w_hi]

print(f'image patch: shape={img_patch.shape} mean={img_patch.mean():.3f} std={img_patch.std():.3f}')
print(f'GT patch FG1: {(gt_patch==1).sum()}, FG2: {(gt_patch==2).sum()}')

# === 上 GPU, 简化 PlainConvUNet (与 D096 同款) ===
img_t = torch.from_numpy(img_patch.astype(np.float32)).view(1, 1, *img_patch.shape).to(device)
gt_t = torch.from_numpy(gt_patch.astype(np.int64)).view(1, *gt_patch.shape).to(device)

# build PlainConvUNet (与 plans 一致, 5 stages, 3 输出 classes)
net = PlainConvUNet(
    input_channels=1,
    n_stages=5,
    features_per_stage=[32, 64, 128, 256, 320],
    conv_op=nn.Conv3d,
    kernel_sizes=[[3,3,3]]*5,
    strides=[[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2]],
    n_conv_per_stage=[2]*5,
    n_conv_per_stage_decoder=[2]*4,
    num_classes=3,
    conv_bias=True,
    norm_op=nn.InstanceNorm3d,
    norm_op_kwargs={'eps': 1e-5, 'affine': True},
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={'inplace': True},
    deep_supervision=False,
).to(device)
print(f'\nnet params: {sum(p.numel() for p in net.parameters())/1e6:.1f}M')

opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.99, nesterov=True, weight_decay=3e-5)

# Use stock nnUNet DC+CE loss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
loss_fn = DC_and_CE_loss(
    {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': False},
    {}, weight_ce=1, weight_dice=1, ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss
).to(device)

print(f'\n=== train 200 SGD steps on single (image, GT) ===')
print(f'{"step":>6} {"ce+dice loss":>14} {"argmax_dice_fg1":>16} {"argmax_dice_fg2":>16}')
for step in range(201):
    net.train()
    out = net(img_t)  # (1, 3, 96, 96, 96)
    loss = loss_fn(out, gt_t.unsqueeze(1))  # gt_t needs (B, 1, ...)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 10 == 0:
        with torch.no_grad():
            pred_argmax = out.argmax(dim=1)[0]  # (D, H, W)
            for klass in (1, 2):
                tp = ((pred_argmax == klass) & (gt_t[0] == klass)).sum().item()
                fp = ((pred_argmax == klass) & (gt_t[0] != klass)).sum().item()
                fn = ((pred_argmax != klass) & (gt_t[0] == klass)).sum().item()
                dice = 2 * tp / max(2 * tp + fp + fn, 1)
                if klass == 1: d1 = dice
                else: d2 = dice
        print(f'{step:>6} {float(loss):>14.4f} {d1:>16.4f} {d2:>16.4f}')

print(f'\nDONE. Final argmax dice fg1={d1:.4f}, fg2={d2:.4f}')
print('如果 step 200 dice fg1 > 0.5, 数据/loss/模型完全正常, 锁定 bug 在 trainer 复杂性')
print('如果 step 200 dice 还卡 0, 数据本身有问题 (但 align 视觉看是对的)')
