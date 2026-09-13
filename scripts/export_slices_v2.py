"""v2 切片导出：论文协议的全局强度窗（全卷均值归一 + 全局尺度 c）。

替代逐切片 min-max（会导致 GT 窗随切片漂移、PSNR 被窗噪声封顶）。
窗口: v01 = clip( v / mean(volume) / C , 0, 1 )，C 标定自 train 受试者 p99.9（data/global_window.txt）。
"""
import csv
import glob
import os
import sys

import numpy as np
from scipy.ndimage import shift as ndshift
from scipy.ndimage import zoom

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from prepare_ixi import load_pair  # noqa: E402

C = float(open(os.path.join(ROOT, "data", "global_window.txt")).read().strip())
OUT = os.path.join(ROOT, "data", "selfrdb_format_v3")
print(f"全局窗 C = {C}")


def window01(v: np.ndarray) -> np.ndarray:
    return np.clip(v / max(v.mean(), 1e-6) / C, 0.0, 1.0).astype(np.float32)


def _ncc(a, b, slab):
    a = a[:, :, slab].copy()
    b = b[:, :, slab].copy()
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9
    return float((a * b).sum() / d)


def fine_register(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """同窗跨模态 NCC 细配准：半分辨率粗搜(±2px) + 全分辨率细搜(±1px)。"""
    a = window01(t1)
    b = window01(t2)
    Z = a.shape[2]
    slab = slice(Z // 4, 3 * Z // 4)
    ah = zoom(a, 0.5, order=1)
    bh = zoom(b, 0.5, order=1)
    sh = [0, 0, 0]
    bv = _ncc(ah, bh, slice(slab.start // 2, slab.stop // 2))
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                v = _ncc(ah, ndshift(bh, (dy, dx, dz), order=1),
                         slice(slab.start // 2, slab.stop // 2))
                if v > bv:
                    bv, sh = v, [dy, dx, dz]
    # 全分辨率细搜（半分辨率位移×2 + 全分辨率 ±1）
    b = ndshift(b, (sh[0] * 2, sh[1] * 2, sh[2] * 2), order=1)
    bv = _ncc(a, b, slab)
    fine = [0, 0, 0]
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                v = _ncc(a, ndshift(b, (dy, dx, dz), order=1), slab)
                if v > bv:
                    bv, fine = v, [dy, dx, dz]
    b = ndshift(b, tuple(fine), order=1)
    return b.astype(np.float32)


def main():
    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_resplit.csv"), encoding="utf-8")))
    subs = sorted({r["subject"] for r in man})
    split = {}
    for r in man:
        split[r["subject"]] = r["split"]

    cnt = {}
    rows = ["subject,split,z,t1_path,t2_path"]
    for si, sid in enumerate(subs):
        t1f = glob.glob(os.path.join(ROOT, "data", "ixi_raw", "T1", f"{sid}-T1.nii.gz"))[0]
        t2f = glob.glob(os.path.join(ROOT, "data", "ixi_raw", "T2", f"{sid}-T2.nii.gz"))[0]
        t1, t2, _ = load_pair(t1f, t2f)
        t2 = fine_register(t1, t2)          # 细配准（±6px 内跨模态 NCC 最优）
        t1w, t2w = window01(t1), window01(t2)
        Z = t1.shape[2]
        for z in range(20, min(Z - 20, 120)):
            a = np.rot90(t1w[:, :, z], -1)
            b = np.rot90(t2w[:, :, z], -1)
            if (a > 0.05).mean() < 0.25 or (b > 0.05).mean() < 0.25:
                continue  # 跳过接近空白的层
            a = zoom(a, 0.5, order=1).astype(np.float32)
            b = zoom(b, 0.5, order=1).astype(np.float32)
            sp = split[sid]
            pair = []
            for mod, arr in (("T1", a), ("T2", b)):
                d = os.path.join(OUT, mod, sp)
                os.makedirs(d, exist_ok=True)
                k = cnt.get((mod, sp), 0)
                p = os.path.join(d, f"slice_{k}.npy")
                np.save(p, arr)
                cnt[(mod, sp)] = k + 1
                pair.append(p)
            rows.append(f"{sid},{sp},{z},{pair[0]},{pair[1]}")
        print(f"[{si+1}/{len(subs)}] {sid} done")

    with open(os.path.join(ROOT, "data", "ixi_slices", "manifest_v3.csv"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(rows))
    print("counts:", {f"{m}/{s}": cnt[(m, s)] for m in ("T1", "T2") for s in ("train", "val", "test")})


if __name__ == "__main__":
    main()
