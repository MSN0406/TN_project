import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Convert precomputed crop .npy to nnU-Net raw DatasetXXX format.")
    p.add_argument(
        "--precomputed_dir",
        type=str,
        default="/home/jzhou169/tn_locseg/precomputed_phase2_crop96_pred",
        help="Directory containing per-case folders with image.npy/label.npy",
    )
    p.add_argument(
        "--nnunet_raw_base",
        type=str,
        default="/home/jzhou169/tn_locseg/nnUNet_data/nnUNet_raw",
        help="nnUNet_raw base directory",
    )
    p.add_argument("--dataset_id", type=int, default=97, help="nnU-Net dataset id (e.g., 97 -> Dataset097_*)")
    p.add_argument("--dataset_name", type=str, default="TN_Crop96Pred", help="nnU-Net dataset name suffix")
    p.add_argument("--prefix", type=str, default="TN97", help="Case file prefix in nnU-Net raw")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing target dataset directory")
    return p.parse_args()


def main():
    args = parse_args()
    precomputed_dir = Path(args.precomputed_dir)
    assert precomputed_dir.is_dir(), f"precomputed_dir not found: {precomputed_dir}"

    case_ids_file = precomputed_dir / "case_ids.json"
    assert case_ids_file.is_file(), f"Missing case_ids.json in: {precomputed_dir}"
    case_ids = json.loads(case_ids_file.read_text())
    assert isinstance(case_ids, list) and len(case_ids) > 0, "case_ids.json is empty or invalid"

    dataset_dir = Path(args.nnunet_raw_base) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"

    if dataset_dir.exists():
        if args.overwrite:
            for root, dirs, files in os.walk(dataset_dir, topdown=False):
                for fn in files:
                    Path(root, fn).unlink()
                for dn in dirs:
                    Path(root, dn).rmdir()
            dataset_dir.rmdir()
        else:
            raise FileExistsError(f"Target dataset exists: {dataset_dir}. Use --overwrite to replace it.")

    images_tr.mkdir(parents=True, exist_ok=False)
    labels_tr.mkdir(parents=True, exist_ok=False)

    case_mapping = {}
    affine = np.eye(4, dtype=np.float32)

    for idx, case_id in enumerate(tqdm(case_ids, desc="Converting cases")):
        case_dir = precomputed_dir / case_id
        img_np = np.load(case_dir / "image.npy").astype(np.float32, copy=False)
        lbl_np = np.load(case_dir / "label.npy").astype(np.int16, copy=False)
        lbl_np = np.clip(lbl_np, 0, 2)

        nn_case = f"{args.prefix}_{idx:04d}"
        img_out = images_tr / f"{nn_case}_0000.nii.gz"
        lbl_out = labels_tr / f"{nn_case}.nii.gz"

        nib.save(nib.Nifti1Image(img_np, affine), str(img_out))
        nib.save(nib.Nifti1Image(lbl_np, affine), str(lbl_out))
        case_mapping[nn_case] = case_id

    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "nerve": 1, "vessel": 2},
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    (dataset_dir / "case_mapping.json").write_text(json.dumps(case_mapping, indent=2))

    print(f"[Done] Created: {dataset_dir}")
    print(f"[Done] numTraining={len(case_ids)}")


if __name__ == "__main__":
    main()

