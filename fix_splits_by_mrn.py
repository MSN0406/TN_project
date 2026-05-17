"""
fix_splits_by_mrn.py
====================
修复 Dataset001_TN 和 Dataset096_TN_Crop96 的 splits_final.json,
按 MRN (病人级) 做 5-fold, 杜绝同一病人的 ipsi/contra 跨 train/val.

- 旧 splits 自动备份为 splits_final.json.bak_<timestamp>_pre_mrn_fix
- 用相同的 random_state, 保证 D001 / D096 的 fold-MRN 分配一致
- 跑完打印每个 fold 的 leakage 验证 (overlap 必须为 0)

跑法:
    python fix_splits_by_mrn.py
"""
import os
REPO_ROOT = os.environ.get('TN_ROOT', os.path.dirname(os.path.abspath(__file__)))

import json
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime

ROOT = "${TN_ROOT}"
DATASETS = ["Dataset001_TN", "Dataset096_TN_Crop96"]
SEED = 42  # 固定, 保证 D001 / D096 / 后续重跑都用同一 fold 分配


def get_patient_id(prepared_name: str) -> str:
    """
    从 prepared_data case 名提取 patient ID.
      'XXXXXXXX_ipsi'  / '_contra' / '_left' / '_right'  -> 去掉后缀作为 MRN
      'sub-XXXXX_*'                                       -> 整串作为 patient ID (openneuro)
      其它                                                 -> 整串
    """
    for suffix in ("_ipsi", "_contra", "_left", "_right"):
        if prepared_name.endswith(suffix):
            return prepared_name[: -len(suffix)]
    return prepared_name


def make_kfold_assignment(items, k=5, seed=SEED):
    """简单确定性 KFold (不依赖 sklearn)."""
    items_sorted = sorted(items)
    rng = random.Random(seed)
    shuffled = list(items_sorted)
    rng.shuffle(shuffled)
    n = len(shuffled)
    base = n // k
    rem = n % k
    folds = []
    cursor = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        folds.append(set(shuffled[cursor : cursor + size]))
        cursor += size
    assert sum(len(f) for f in folds) == n
    return folds


def main():
    # ---------------- step 1: 读两个 dataset 的 case_mapping ----------------
    mappings = {}
    for ds in DATASETS:
        path = f"{ROOT}/nnUNet_data/nnUNet_raw/{ds}/case_mapping.json"
        if not os.path.isfile(path):
            sys.exit(f"missing: {path}")
        mappings[ds] = json.load(open(path))

    # 两个 mapping 的 prepared_name 集合应该完全一致 (同一批病人)
    sets = {ds: set(m.values()) for ds, m in mappings.items()}
    if sets[DATASETS[0]] != sets[DATASETS[1]]:
        diff = sets[DATASETS[0]] ^ sets[DATASETS[1]]
        sys.exit(f"prepared sets differ between {DATASETS}: {len(diff)} cases differ")
    prepared_all = sets[DATASETS[0]]
    print(f"prepared cases (共享): {len(prepared_all)}")

    # ---------------- step 2: 按 MRN 分组 ----------------
    prepared_by_pid = defaultdict(list)
    for prep in prepared_all:
        prepared_by_pid[get_patient_id(prep)].append(prep)
    pids = sorted(prepared_by_pid.keys())
    sizes = [len(prepared_by_pid[p]) for p in pids]
    print(f"unique patient IDs: {len(pids)}")
    print(f"  双侧病人 (ipsi+contra): {sum(1 for s in sizes if s >= 2)}")
    print(f"  单侧病人:               {sum(1 for s in sizes if s == 1)}")
    print(f"  prepared cases / patient: min={min(sizes)} max={max(sizes)}")

    # ---------------- step 3: 在 patient 级做 5-fold ----------------
    val_folds_by_pid = make_kfold_assignment(pids, k=5, seed=SEED)
    print(f"\n5-fold (patient-level) sizes: "
          f"{[len(f) for f in val_folds_by_pid]} (val MRNs per fold)")

    # ---------------- step 4: 对每个 dataset, 写入新的 splits_final.json ----------------
    print()
    for ds in DATASETS:
        case_mapping = mappings[ds]
        prepared_to_case = {v: k for k, v in case_mapping.items()}
        all_pids_set = set(pids)

        new_splits = []
        for fi, val_pids in enumerate(val_folds_by_pid):
            train_pids = all_pids_set - val_pids

            train_cases = sorted(
                prepared_to_case[p]
                for pid in train_pids
                for p in prepared_by_pid[pid]
                if p in prepared_to_case
            )
            val_cases = sorted(
                prepared_to_case[p]
                for pid in val_pids
                for p in prepared_by_pid[pid]
                if p in prepared_to_case
            )
            new_splits.append({"train": train_cases, "val": val_cases})

        splits_path = f"{ROOT}/nnUNet_data/nnUNet_preprocessed/{ds}/splits_final.json"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{splits_path}.bak_{ts}_pre_mrn_fix"
        if os.path.isfile(splits_path):
            shutil.copy(splits_path, backup_path)
        with open(splits_path, "w") as f:
            json.dump(new_splits, f, indent=2)

        print(f"[{ds}]")
        print(f"  backup -> {os.path.basename(backup_path)}")
        print(f"  wrote  -> {splits_path}")
        for fi, fold in enumerate(new_splits):
            tr_pids = set(get_patient_id(case_mapping[c]) for c in fold["train"])
            vl_pids = set(get_patient_id(case_mapping[c]) for c in fold["val"])
            ov = tr_pids & vl_pids
            tag = "✅" if not ov else "⚠️ LEAK"
            print(f"  fold {fi}: train={len(fold['train']):4d} val={len(fold['val']):3d} "
                  f"| train_MRNs={len(tr_pids):3d} val_MRNs={len(vl_pids):3d} "
                  f"OVERLAP={len(ov):2d} {tag}")
        print()


if __name__ == "__main__":
    main()
