"""把 data/synthrad_pelvis/<pid>/{mr,ct,mask}.nii.gz 切成切片 npy + manifest。

约定（与官方 create_brats_dataset.py 一致）：逐切片 min-max → [0,1]。
输出: data/synthrad_pelvis_slices/<pid>_z<NNN>_{mr,ct,mask}.npy + manifest.csv
"""
import argparse
import csv
import os

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "synthrad_pelvis")
OUT = os.path.join(ROOT, "data", "synthrad_pelvis_slices")


def minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-slices", type=int, default=80, help="每患者最多切片数(取有身体覆盖的中间段)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    rows = []
    pids = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))
    print(f"患者: {pids}")
    for pid in pids:
        mr_p = os.path.join(SRC, pid, "mr.nii.gz")
        ct_p = os.path.join(SRC, pid, "ct.nii.gz")
        mk_p = os.path.join(SRC, pid, "mask.nii.gz")
        if not (os.path.exists(mr_p) and os.path.exists(ct_p)):
            print(f"  跳过 {pid}（文件不全）")
            continue
        mr_img = nib.load(mr_p)
        ct_img = nib.load(ct_p)
        mr = mr_img.get_fdata().astype(np.float32)
        ct = ct_img.get_fdata().astype(np.float32)
        if ct.shape != mr.shape:
            from nibabel.processing import resample_from_to
            ct = resample_from_to(ct_img, mr_img).get_fdata().astype(np.float32)
            print(f"  {pid}: CT 重采样 {ct.shape} 对齐 MR")
        mk = nib.load(mk_p).get_fdata().astype(np.float32) if os.path.exists(mk_p) \
            else np.ones_like(mr)
        if mk.shape != mr.shape:
            from nibabel.processing import resample_from_to
            mk = resample_from_to(nib.load(mk_p), mr_img, order=0).get_fdata().astype(np.float32)

        Z = mr.shape[2]
        cover = [(z, float((mk[:, :, z] > 0).mean())) for z in range(Z)]
        good = [z for z, c in cover if c > 0.2]
        if len(good) > args.max_slices:
            mid = len(good) // 2
            half = args.max_slices // 2
            good = good[mid - half: mid + half]
        for z in good:
            base = f"{pid}_z{z:03d}"
            np.save(os.path.join(OUT, base + "_mr.npy"), minmax(mr[:, :, z]))
            np.save(os.path.join(OUT, base + "_ct.npy"), minmax(ct[:, :, z]))
            np.save(os.path.join(OUT, base + "_mask.npy"), (mk[:, :, z] > 0).astype(np.uint8))
            rows.append({
                "patient": pid, "z": z,
                "mr_path": os.path.join("data", "synthrad_pelvis_slices", base + "_mr.npy"),
                "ct_path": os.path.join("data", "synthrad_pelvis_slices", base + "_ct.npy"),
                "mask_path": os.path.join("data", "synthrad_pelvis_slices", base + "_mask.npy"),
                "body_frac": round(float((mk[:, :, z] > 0).mean()), 4),
            })
        print(f"  {pid}: {len(good)} 片 (shape {mr.shape[:2]})")

    with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"共 {len(rows)} 片 -> {OUT}")


if __name__ == "__main__":
    main()
