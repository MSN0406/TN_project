"""
diag_visual_alignment.py
========================
跟训练一模一样地走 phase3 的 image crop 和 GT crop 路径, 把结果画成 PNG 看是否对齐.

要回答的问题:
  - 图像 patch 中的 TN 在哪 (高亮亚组织区域)
  - GT patch 中的 nerve (class 1) 标签在哪
  - 它们俩对齐吗

输出: ${TN_ROOT}/diag_outputs/align_TN0004_*.png
  每张图分 3 行: 9 个切片 (3 axial + 3 coronal + 3 sagittal)
  每个切片三联: image | GT mask | image+GT overlay
"""
import os
REPO_ROOT = os.environ.get('TN_ROOT', os.path.dirname(os.path.abspath(__file__)))

import json, os, sys
sys.path.insert(0, REPO_ROOT)

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.differentiable_crop import differentiable_crop_3d

OUT = os.path.join(REPO_ROOT, 'diag_outputs')
os.makedirs(OUT, exist_ok=True)

PREP = os.path.join(REPO_ROOT, 'nnUNet_data/nnUNet_preprocessed/Dataset001_TN')
PREPARED = os.path.join(REPO_ROOT, 'prepared_data')
case_mapping = json.load(open(os.path.join(REPO_ROOT, 'nnUNet_data/nnUNet_raw/Dataset001_TN/case_mapping.json')))

NNUNET_KEY = 'TN_0004'
prep_name = case_mapping[NNUNET_KEY]
print(f'{NNUNET_KEY} -> {prep_name}')

# === STEP 1: load wb + apply per-case z-score (matches #24 fix) ===
info = json.load(open(f'{PREPARED}/{prep_name}/info.json'))
wb_nii = nib.load(info['nii_path'])
wb = wb_nii.get_fdata().astype(np.float32)
if wb.ndim == 4: wb = wb[..., 0]
spacing_prep = np.array(wb_nii.header.get_zooms()[:3], dtype=np.float64)
print(f'wb native: {wb.shape}, spacing: {spacing_prep}')

# resample to 1mm (TN_WB_TARGET_SPACING=1,1,1)
target_spacing = np.array([1.0, 1.0, 1.0])
out_shape = tuple(int(x) for x in np.maximum(1, np.round(np.array(wb.shape) * spacing_prep / target_spacing)))
print(f'wb resampled: {out_shape}')
wb_t = torch.from_numpy(wb).view(1, 1, *wb.shape)
wb_t = F.interpolate(wb_t, size=out_shape, mode='trilinear', align_corners=False)
wb = wb_t[0, 0].numpy()

# per-case z-score (matches #24 fix, full-image)
mean = float(wb.mean()); std = float(wb.std())
wb = (wb - mean) / (std + 1e-8)
print(f'after z-score: mean={wb.mean():.3f} std={wb.std():.3f} min={wb.min():.3f} max={wb.max():.3f}')

# === STEP 2: GT centroid in resampled space ===
centroid_native = np.load(f'{PREPARED}/{prep_name}/centroid.npy').astype(np.float64)
gt_centroid_resampled = centroid_native * (spacing_prep / target_spacing)
print(f'centroid in resampled space: {gt_centroid_resampled}')
wb_shape = np.array(wb.shape, dtype=np.float32)
gt_centroid_norm = gt_centroid_resampled / np.maximum(wb_shape - 1.0, 1.0)
print(f'centroid_norm in [0,1]: {gt_centroid_norm}')

# === STEP 3: 同 trainer 的 image crop ===
PATCH = 96
wb_tensor = torch.from_numpy(wb).view(1, 1, *wb.shape)
center_norm_t = torch.from_numpy(gt_centroid_norm.astype(np.float32)).view(1, 3)

# round centroid to integer voxel (mimics _round_centroid_to_voxel)
scale = torch.tensor([max(s - 1, 1) for s in wb.shape], dtype=torch.float32)
center_vox_t = torch.round(center_norm_t * scale)
center_vox_t = torch.clamp(center_vox_t, torch.zeros(3), scale)
center_norm_rounded = center_vox_t / scale

img_patch = differentiable_crop_3d(
    wb_tensor, center_norm_rounded, PATCH, mode='bilinear', padding_mode='border'
)
img_patch = img_patch[0, 0].numpy()
print(f'\nimg patch: shape={img_patch.shape}, mean={img_patch.mean():.3f} std={img_patch.std():.3f}')

# === STEP 4: 同 trainer 的 GT crop (新版 #25 fix) ===
# load GT and embed at native shape, then resample to wb_shape — 同 _load_gt_seg_resized_to_wb
seg_nii = nib.load(f'{PREP}/gt_segmentations/{NNUNET_KEY}.nii.gz')
seg = seg_nii.get_fdata()
if seg.ndim == 4: seg = seg[..., 0]
seg = np.rint(seg).astype(np.int64)
spacing_nnu = np.array(seg_nii.header.get_zooms()[:3], dtype=np.float64)
target_shape = np.maximum(1, np.round(np.array(seg.shape) * spacing_nnu / spacing_prep).astype(int))
seg_t = torch.from_numpy(seg.astype(np.float32)).view(1, 1, *seg.shape)
seg_t = F.interpolate(seg_t, size=tuple(int(x) for x in target_shape), mode='nearest')
seg_resamp = torch.round(seg_t[0, 0]).long().numpy()

native_shape = tuple(int(x) for x in nib.load(info['nii_path']).shape[:3])
center_native = np.round(centroid_native).astype(np.int64)
half = target_shape // 2
full_native = np.zeros(native_shape, dtype=np.int64)

def emb(c, h, t, dim):
    lo_full = c - h; hi_full = lo_full + t
    lo_clip = max(0, int(lo_full)); hi_clip = min(int(dim), int(hi_full))
    if hi_clip <= lo_clip: return None, None
    src_lo = lo_clip - int(lo_full); src_hi = src_lo + (hi_clip - lo_clip)
    return slice(lo_clip, hi_clip), slice(src_lo, src_hi)

s0, ss0 = emb(center_native[0], half[0], target_shape[0], native_shape[0])
s1, ss1 = emb(center_native[1], half[1], target_shape[1], native_shape[1])
s2, ss2 = emb(center_native[2], half[2], target_shape[2], native_shape[2])
full_native[s0, s1, s2] = seg_resamp[ss0, ss1, ss2]

if native_shape != out_shape:
    full_t = torch.from_numpy(full_native.astype(np.float32)).view(1, 1, *native_shape)
    full_t = F.interpolate(full_t, size=out_shape, mode='nearest')
    seg_full = torch.round(full_t[0, 0]).long().numpy()
else:
    seg_full = full_native
print(f'\nseg_full at wb_shape: shape={seg_full.shape}, FG1={(seg_full==1).sum()}, FG2={(seg_full==2).sum()}')

# Now apply NEW #25 GT crop (centered + OOB pad with 0)
center_vox_int = center_vox_t.long()[0].numpy()
print(f'center voxel (rounded): {center_vox_int}, half_patch={PATCH//2}')

halves = [PATCH // 2, PATCH // 2, PATCH // 2]
crops = [PATCH, PATCH, PATCH]
case_dims = wb.shape

lo_raw = [int(center_vox_int[d]) - halves[d] for d in range(3)]
hi_raw = [lo_raw[d] + crops[d] for d in range(3)]
print(f'GT crop window (raw): lo={lo_raw}, hi={hi_raw}')
print(f'  case_dims: {case_dims}')

gt_patch = np.zeros((PATCH, PATCH, PATCH), dtype=np.int64)
d_lo = max(0, lo_raw[0]); d_hi = min(case_dims[0], hi_raw[0])
h_lo = max(0, lo_raw[1]); h_hi = min(case_dims[1], hi_raw[1])
w_lo = max(0, lo_raw[2]); w_hi = min(case_dims[2], hi_raw[2])
pd_lo = d_lo - lo_raw[0]; pd_hi = pd_lo + (d_hi - d_lo)
ph_lo = h_lo - lo_raw[1]; ph_hi = ph_lo + (h_hi - h_lo)
pw_lo = w_lo - lo_raw[2]; pw_hi = pw_lo + (w_hi - w_lo)
print(f'  intersection in case: [{d_lo}:{d_hi}, {h_lo}:{h_hi}, {w_lo}:{w_hi}]')
print(f'  target offset in patch: [{pd_lo}:{pd_hi}, {ph_lo}:{ph_hi}, {pw_lo}:{pw_hi}]')

src = seg_full[d_lo:d_hi, h_lo:h_hi, w_lo:w_hi]
gt_patch[pd_lo:pd_hi, ph_lo:ph_hi, pw_lo:pw_hi] = src
print(f'\ngt_patch FG1: {(gt_patch == 1).sum()}, FG2: {(gt_patch == 2).sum()}')

# === STEP 5: 画图 ===
def fg_centroid(p, klass):
    mask = (p == klass)
    if mask.sum() == 0: return None
    coords = np.argwhere(mask).astype(np.float32)
    return coords.mean(axis=0).astype(int)

cent_fg1 = fg_centroid(gt_patch, 1)
print(f'\nFG1 centroid in GT patch: {cent_fg1} (expected ~ patch center [48,48,48])')

# 取 image patch FG1 GT centroid 所在切片 (如果 GT 有 FG1, 否则 patch 中心)
if cent_fg1 is not None:
    sd, sh, sw = cent_fg1
else:
    sd = sh = sw = PATCH // 2
print(f'Slicing at (D={sd}, H={sh}, W={sw})')

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
slice_specs = [
    ('axial (D)', img_patch[sd, :, :], gt_patch[sd, :, :]),
    ('coronal (H)', img_patch[:, sh, :], gt_patch[:, sh, :]),
    ('sagittal (W)', img_patch[:, :, sw], gt_patch[:, :, sw]),
]
for r, (title, img2d, gt2d) in enumerate(slice_specs):
    img_norm = (img2d - img2d.min()) / (img2d.max() - img2d.min() + 1e-8)
    axes[r, 0].imshow(img_norm, cmap='gray')
    axes[r, 0].set_title(f'{title} - image patch')
    cy, cx = np.array(img2d.shape) // 2
    axes[r, 0].axhline(cy, color='lime', lw=0.5, alpha=0.5)
    axes[r, 0].axvline(cx, color='lime', lw=0.5, alpha=0.5)

    # GT mask: red=class1 (nerve), blue=class2 (vessel)
    gt_rgb = np.zeros((*gt2d.shape, 3))
    gt_rgb[gt2d == 1] = [1, 0, 0]
    gt_rgb[gt2d == 2] = [0, 0, 1]
    axes[r, 1].imshow(gt_rgb)
    axes[r, 1].set_title(f'{title} - GT (red=nerve, blue=vessel)')
    axes[r, 1].axhline(cy, color='lime', lw=0.5, alpha=0.5)
    axes[r, 1].axvline(cx, color='lime', lw=0.5, alpha=0.5)

    # overlay
    overlay = np.stack([img_norm]*3, axis=-1)
    overlay[gt2d == 1] = [1, 0.3, 0.3]
    overlay[gt2d == 2] = [0.3, 0.3, 1]
    axes[r, 2].imshow(overlay)
    axes[r, 2].set_title(f'{title} - overlay (lime cross = patch center)')
    axes[r, 2].axhline(cy, color='lime', lw=0.5, alpha=0.5)
    axes[r, 2].axvline(cx, color='lime', lw=0.5, alpha=0.5)

for ax in axes.flat:
    ax.axis('off')
plt.suptitle(
    f'TN_0004 phase3 alignment check\n'
    f'centroid_norm={gt_centroid_norm.round(3).tolist()} -> voxel {center_vox_int.tolist()}\n'
    f'GT FG1 centroid in patch: {cent_fg1.tolist() if cent_fg1 is not None else "NONE"}\n'
    f'IF aligned: nerve (red) should sit on lime crosshair (patch center).\n'
    f'IF misaligned: nerve will be offset from center.',
    fontsize=11
)
plt.tight_layout()
out_path = f'{OUT}/align_TN0004.png'
plt.savefig(out_path, dpi=80, bbox_inches='tight')
plt.close()
print(f'\nsaved -> {out_path}')

# Also save raw stats
with open(f'{OUT}/align_TN0004.txt', 'w') as f:
    f.write(f'TN_0004 alignment stats\n')
    f.write(f'wb_shape: {wb.shape}\n')
    f.write(f'centroid voxel: {center_vox_int.tolist()}\n')
    f.write(f'patch size: {PATCH}\n')
    f.write(f'expected centroid in patch: ~[{PATCH//2}, {PATCH//2}, {PATCH//2}]\n')
    f.write(f'GT FG1 centroid in patch: {cent_fg1.tolist() if cent_fg1 is not None else None}\n')
    f.write(f'GT FG1 voxels: {(gt_patch==1).sum()}\n')
    f.write(f'GT FG2 voxels: {(gt_patch==2).sum()}\n')
    f.write(f'image patch stats: min={img_patch.min():.3f} max={img_patch.max():.3f} mean={img_patch.mean():.3f}\n')
print(f'saved -> {OUT}/align_TN0004.txt')
