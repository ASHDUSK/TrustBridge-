"""CT 自训练数据导出：SynthRAD 盆腔 → SelfRDB 目录格式（128²，垫方保比例）。

划分（患者级，无重叠）：
  center A（T1w→CT，7 患者）: train 5 / val 1 / test 1
  center C（T2w→CT，3 患者）: train 2 / val(=train 末 20 片，含警示) / test 1

输出:
  data/ct_a_format/{T1,CT}/{train,val,test}/slice_*.npy
  data/ct_c_format/{T2,CT}/{train,val,test}/slice_*.npy
  data/synthrad_slices_128/manifest_ct_a.csv, manifest_ct_c.csv（eval 用）
"""
import csv
import glob
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S128 = 128


def pad_square_resize(img: np.ndarray) -> np.ndarray:
    """垫方（居中）→ 128×128（双线性）。"""
    H, W = img.shape
    S = max(H, W)
    canvas = np.zeros((S, S), dtype=np.float32)
    top, left = (S - H) // 2, (S - W) // 2
    canvas[top:top + H, left:left + W] = img
    im = Image.fromarray((canvas * 255).astype(np.uint8))
    im = im.resize((S128, S128), Image.BILINEAR)
    return np.asarray(im).astype(np.float32) / 255.0


def export_center(center: str, mr_tag: str, patients: dict, manifest: str):
    """patients: {patient: split}；manifest: 官方清单（mr/ct 路径）。"""
    rows = list(csv.DictReader(open(os.path.join(ROOT, manifest), encoding="utf-8")))
    out_dir = os.path.join(ROOT, f"data/ct_{center.lower()}_format")
    ev_dir = os.path.join(ROOT, "data", "synthrad_slices_128")
    os.makedirs(ev_dir, exist_ok=True)
    ev_rows = ["patient,split,z,mr_path,ct_path"]
    cnt = {}
    for r in rows:
        pat = r["patient"]
        split = patients[pat]
        z = int(r["z"])
        mr = np.load(os.path.join(ROOT, r["mr_path"])).astype(np.float32)
        ct = np.load(os.path.join(ROOT, r["ct_path"])).astype(np.float32)
        mr128 = pad_square_resize(mr)
        ct128 = pad_square_resize(ct)
        pair = []
        for mod, arr in ((mr_tag, mr128), ("CT", ct128)):
            d = os.path.join(out_dir, mod, split)
            os.makedirs(d, exist_ok=True)
            k = cnt.get((mod, split), 0)
            p = os.path.join(d, f"slice_{k}.npy")
            np.save(p, arr)
            cnt[(mod, split)] = k + 1
            pair.append(p)
        if split in ("val", "test"):
            ev_rows.append(f"{pat},{split},{z},{pair[0]},{pair[1]}")
    with open(os.path.join(ev_dir, f"manifest_ct_{center.lower()}.csv"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(ev_rows))
    print(f"center {center}: ", {f"{m}/{s}": cnt.get((m, s), 0) for m in (mr_tag, "CT")
                                 for s in ("train", "val", "test")})


def main():
    export_center("A", "T1",
                  {"1PA001": "train", "1PA030": "train", "1PA062": "train",
                   "1PA094": "train", "1PA117": "train",
                   "1PA148": "val", "1PA173": "test"},
                  "data/synthrad_pelvis_slices/manifest_centerA.csv")
    export_center("C", "T2",
                  {"1PC011": "train", "1PC040": "train", "1PC069": "test"},
                  "data/synthrad_pelvis_slices/manifest_centerC.csv")
    # C 中心只有 3 患者：val 用 train 患者末 20 片（同患者泄露，已文档化）
    d = os.path.join(ROOT, "data/ct_c_format")
    for mod in ("T2", "CT"):
        src = sorted(glob.glob(os.path.join(d, mod, "train", "slice_*.npy")))
        val_dir = os.path.join(d, mod, "val")
        os.makedirs(val_dir, exist_ok=True)
        for k, p in enumerate(src[-20:]):
            os.replace(p, os.path.join(val_dir, f"slice_{k}.npy"))
        remain = sorted(glob.glob(os.path.join(d, mod, "train", "slice_*.npy")))
        for k, p in enumerate(remain):
            os.replace(p, os.path.join(d, mod, "train", f"slice_{k}.npy"))
    print("C 中心 val = train 患者末 20 片（同患者泄露已文档化）")


if __name__ == "__main__":
    main()
