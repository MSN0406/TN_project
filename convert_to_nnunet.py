"""
convert_to_nnunet.py
====================
将 prepared_data/ 转换为 nnU-Net v2 数据集格式.

对每个 case:
  1. 从全脑 NIfTI 裁剪 64x64x64 ROI (以 GT centroid 为中心)
  2. 保存 raw crop 为 imagesTr/  (不做归一化, 让 nnU-Net 自己处理)
  3. 保存 mask 为 labelsTr/

用法:
  python convert_to_nnunet.py
"""
import os
REPO_ROOT = os.environ.get('TN_ROOT', os.path.dirname(os.path.abspath(__file__)))

import json
import os
import shutil

import nibabel as nib
import numpy as np


PREPARED_DIR = os.path.join(REPO_ROOT, 'prepared_data')
NNUNET_RAW = os.path.join(REPO_ROOT, 'nnUNet_data/nnUNet_raw')
DATASET_ID = 1
DATASET_NAME = "Dataset001_TN"
CROP_SIZE = 64


def center_crop_from_brain(brain_data, centroid, crop_size=64):
    """从全脑提取以 centroid 为中心的 crop, 边界处做 padding."""
    half = crop_size // 2
    d, h, w = brain_data.shape[:3]

    c = np.round(centroid).astype(int)

    starts = c - half
    ends = starts + crop_size

    # clamp
    src_starts = np.maximum(starts, 0)
    src_ends = np.minimum(ends, np.array([d, h, w]))

    dst_starts = src_starts - starts
    dst_ends = dst_starts + (src_ends - src_starts)

    crop = np.zeros((crop_size, crop_size, crop_size), dtype=brain_data.dtype)
    crop[dst_starts[0]:dst_ends[0],
         dst_starts[1]:dst_ends[1],
         dst_starts[2]:dst_ends[2]] = brain_data[
             src_starts[0]:src_ends[0],
             src_starts[1]:src_ends[1],
             src_starts[2]:src_ends[2]]

    return crop


def main():
    dataset_dir = os.path.join(NNUNET_RAW, DATASET_NAME)
    images_dir = os.path.join(dataset_dir, "imagesTr")
    labels_dir = os.path.join(dataset_dir, "labelsTr")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    with open(os.path.join(PREPARED_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    case_ids = metadata["case_ids"]
    print(f"Total cases: {len(case_ids)}")

    wb_cache = {}
    success = 0
    errors = []

    for i, cid in enumerate(case_ids):
        case_dir = os.path.join(PREPARED_DIR, cid)
        info_path = os.path.join(case_dir, "info.json")
        if not os.path.isfile(info_path):
            errors.append((cid, "info.json not found"))
            continue

        with open(info_path) as f:
            info = json.load(f)

        # nnU-Net case naming: TN_XXXX
        case_name = f"TN_{i:04d}"

        try:
            # 1. Load whole brain and crop
            nii_path = info["nii_path"]
            if nii_path not in wb_cache:
                nii = nib.load(nii_path)
                wb = nii.get_fdata().astype(np.float32)
                if wb.ndim == 4:
                    wb = wb[:, :, :, 0]
                wb_cache[nii_path] = wb
            else:
                wb = wb_cache[nii_path]

            centroid = np.load(os.path.join(case_dir, "centroid.npy"))
            crop = center_crop_from_brain(wb, centroid, CROP_SIZE)

            # 2. Save image crop (raw, no normalization)
            img_nii = nib.Nifti1Image(crop, affine=np.eye(4))
            nib.save(img_nii, os.path.join(images_dir, f"{case_name}_0000.nii.gz"))

            # 3. Save mask
            mask_path = os.path.join(case_dir, "mask_aligned.nii.gz")
            mask_nii = nib.load(mask_path)
            mask = mask_nii.get_fdata().astype(np.uint8)
            mask_out = nib.Nifti1Image(mask, affine=np.eye(4))
            nib.save(mask_out, os.path.join(labels_dir, f"{case_name}.nii.gz"))

            success += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(case_ids)}] done")

        except Exception as e:
            errors.append((cid, str(e)))

    # 4. Create dataset.json
    dataset_json = {
        "channel_names": {
            "0": "MRI"
        },
        "labels": {
            "background": 0,
            "nerve": 1,
            "vessel": 2
        },
        "numTraining": success,
        "file_ending": ".nii.gz"
    }

    with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    # 5. Save case_id mapping for later reference
    mapping = {}
    idx = 0
    for cid in case_ids:
        case_dir = os.path.join(PREPARED_DIR, cid)
        if os.path.isfile(os.path.join(case_dir, "info.json")):
            mapping[f"TN_{idx:04d}"] = cid
            idx += 1

    with open(os.path.join(dataset_dir, "case_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\nDone! {success} cases converted, {len(errors)} errors")
    if errors:
        for cid, err in errors[:10]:
            print(f"  Error: {cid}: {err}")
    print(f"\nDataset dir: {dataset_dir}")
    print(f"dataset.json created")


if __name__ == "__main__":
    main()
