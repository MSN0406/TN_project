"""分析验证集 GT mask 质量, 筛选 nerve+vessel 都有的 case."""
import yaml, os
import nibabel as nib
import numpy as np
from data.dataset import get_train_val_split

with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

_, val_ids = get_train_val_split(
    cfg["data"]["prepared_dir"],
    train_ratio=cfg["data"]["train_val_split"],
    seed=cfg["data"]["random_seed"],
)

header = "{:<25} {:>10} {:>11} {:>5}".format("case_id", "nerve_vox", "vessel_vox", "both")
print(header)
print("-" * 55)

good_cases = []
for cid in sorted(val_ids):
    mask_path = os.path.join(cfg["data"]["prepared_dir"], cid, "mask_aligned.nii.gz")
    if not os.path.exists(mask_path):
        continue
    mask = nib.load(mask_path).get_fdata().astype(int)
    # center crop to 48
    d, h, w = mask.shape
    c = 48
    d0, h0, w0 = (d - c) // 2, (h - c) // 2, (w - c) // 2
    crop = mask[d0:d0+c, h0:h0+c, w0:w0+c]

    nerve = int((crop == 1).sum())
    vessel = int((crop == 2).sum())
    both = "Y" if nerve > 0 and vessel > 0 else "N"
    print("{:<25} {:>10} {:>11} {:>5}".format(cid, nerve, vessel, both))
    if nerve > 50 and vessel > 50:
        good_cases.append(cid)

print("\n总验证集: {} cases".format(len(val_ids)))
print("两类都有 (>50 vox): {} cases".format(len(good_cases)))
print("\nGood cases:")
for c in good_cases:
    print("  " + c)
