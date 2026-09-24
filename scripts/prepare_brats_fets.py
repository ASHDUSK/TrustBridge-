"""从 brats_sliced.tar.gz（FeTS21 切片包，h5: case + label）抽取选定受试者的切片。

输出:
  data/brats_fets_slices/<SUBJ>_z<NNN>.npy        FLAIR 图像 float32 [0,1] (240x240)
  data/brats_fets_slices/<SUBJ>_z<NNN>_label.npy 肿瘤标注 uint8 (0/1/2/4)
  data/brats_fets_slices/manifest.csv

注意：FeTS21 切片包仅含单一模态图像(case, 经验判断为 FLAIR) + 肿瘤标注，
无法提供 T2/T1 配对 GT —— 本数据用于"信任图 vs 肿瘤标注"分析，不做 PSNR。
"""
import argparse
import csv
import io
import os
import sys
import tarfile

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAR = os.path.join(ROOT, "data", "brats_sliced.tar.gz")
OUT = os.path.join(ROOT, "data", "brats_fets_slices")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=329, help="起始受试者编号(含)")
    ap.add_argument("--last", type=int, default=340, help="结束受试者编号(含)")
    args = ap.parse_args()

    subjects = {f"FeTS21_Training_{i:03d}" for i in range(args.first, args.last + 1)}
    os.makedirs(OUT, exist_ok=True)
    rows = []
    kept = 0

    tf = tarfile.open(TAR, "r:gz")
    for m in tf:
        if not m.name.endswith(".h5"):
            continue
        stem = os.path.basename(m.name)[:-3]            # slice_0_FeTS21_Training_001
        parts = stem.split("_")
        z = int(parts[1])
        subj = "_".join(parts[2:])                      # FeTS21_Training_001
        if subj not in subjects:
            continue
        buf = io.BytesIO(tf.extractfile(m).read())
        with h5py.File(buf, "r") as h:
            case = h["case"][:].astype(np.float32) / 255.0
            label = h["label"][:].astype(np.uint8)
        base = f"{subj}_z{z:03d}"
        np.save(os.path.join(OUT, base + ".npy"), case)
        np.save(os.path.join(OUT, base + "_label.npy"), label)
        rows.append({
            "subject": subj, "z": z,
            "case_path": os.path.join("data", "brats_fets_slices", base + ".npy"),
            "label_path": os.path.join("data", "brats_fets_slices", base + "_label.npy"),
            "tumor_frac": round(float((label > 0).mean()), 5),
            "brain_frac": round(float((case > 0.02).mean()), 5),
        })
        kept += 1
        if kept % 300 == 0:
            print(f"  已抽取 {kept} 切片 ...")

    with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["subject"], r["z"])))

    tumor_slices = [r for r in rows if r["tumor_frac"] > 0.003]
    print(f"完成: {kept} 切片 / {len(subjects)} 受试者; 含肿瘤(>0.3%面积)切片 {len(tumor_slices)}")
    print(f"manifest -> {os.path.join(OUT, 'manifest.csv')}")


if __name__ == "__main__":
    main()
