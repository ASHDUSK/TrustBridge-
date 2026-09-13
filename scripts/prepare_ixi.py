"""从 IXI tar 包准备配对切片数据集（供应用演示与信任图验证）。

流程：
  1. 解包 data/IXI-{T1,T2}-first800.tar（截断 tar：只提取完整成员）
  2. 按受试者 ID 配对 T1/T2
  3. T2 头文件仿射（世界坐标）重采样到 T1 网格（退化时按形状比例回退）
  4. 逐体归一化 → 提取含脑轴位切片（≤100 层/人）→ 256×256 [0,1]
  5. 输出 npy 对 + manifest.csv + 演示示例 PNG

用法: python scripts/prepare_ixi.py [--n-val 10] [--max-sub 40]
"""
import argparse
import os
import sys
import tarfile
import numpy as np
import nibabel as nib
from scipy.ndimage import affine_transform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "ixi_raw")
OUT = os.path.join(DATA, "ixi_slices")
EX = os.path.join(ROOT, "app", "examples")


def untar_partial(tar_path: str, dest: str):
    os.makedirs(dest, exist_ok=True)
    n = 0
    with tarfile.open(tar_path) as tf:
        while True:
            try:
                m = tf.next()
            except (tarfile.ReadError, EOFError):
                break
            if m is None:
                break
            if not m.isfile() or not m.name.endswith(".nii.gz"):
                continue
            try:
                tf.extract(m, dest)
                n += 1
            except (tarfile.ReadError, EOFError, OSError):
                break  # 截断文件：到此为止
    print(f"  {os.path.basename(tar_path)} -> {n} volumes -> {dest}")


def subject_id(fname: str) -> str:
    # IXI002-Guys-0828-T1.nii.gz -> IXI002-Guys-0828
    base = os.path.basename(fname)
    parts = base.rsplit("-", 1)
    return parts[0] if len(parts) == 2 else base.split(".")[0]


def load_pair(t1_path: str, t2_path: str):
    t1_img, t2_img = nib.load(t1_path), nib.load(t2_path)
    t1 = t1_img.get_fdata(dtype=np.float32)
    t2v = t2_img.get_fdata(dtype=np.float32)

    def ok_affine(img):
        a = img.affine
        return img.header["sform_code"] > 0 and abs(np.linalg.det(a[:3, :3])) > 1e-6

    if ok_affine(t1_img) and ok_affine(t2_img):
        # 世界坐标对齐：ref体素 i -> 世界 -> src体素 j
        M = np.linalg.inv(t2_img.affine[:3, :3]) @ t1_img.affine[:3, :3]
        off = np.linalg.inv(t2_img.affine[:3, :3]) @ (t1_img.affine[:3, 3] - t2_img.affine[:3, 3])
        t2r = affine_transform(t2v, matrix=M, offset=off,
                               output_shape=t1.shape, order=1, mode="constant")
        method = "affine"
    else:
        # 退化回退：按形状比例缩放
        from scipy.ndimage import zoom
        factors = [t1.shape[i] / t2v.shape[i] for i in range(3)]
        t2r = zoom(t2v, factors, order=1)
        method = "zoom-fallback"
    return t1, t2r.astype(np.float32), method


def norm01(vol: np.ndarray) -> np.ndarray:
    """脑组织均值归一 → 99.5 分位拉伸 → 裁剪 [0,1]。"""
    brain = vol[vol > np.percentile(vol, 40)]
    mu = max(brain.mean(), 1e-6)
    v = vol / mu
    hi = np.percentile(v[v > 0.05], 99.5) if (v > 0.05).any() else 1.0
    return np.clip(v / max(hi, 1e-6), 0.0, 1.0)


def pick_slices(t1v: np.ndarray, t2v: np.ndarray, max_slices=100, min_frac=0.03):
    """选含脑轴位切片：脑体素占比 > min_frac，取中间最多 max_slices 层。"""
    idx = [k for k in range(t1v.shape[2])
           if (t1v[:, :, k] > 0.08).mean() > min_frac
           and (t2v[:, :, k] > 0.08).mean() > min_frac]
    if not idx:
        return []
    mid = len(idx) // 2
    half = max_slices // 2
    sel = idx[max(0, mid - half): mid + half]
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=10)
    ap.add_argument("--max-sub", type=int, default=40)
    ap.add_argument("--slices-per-sub", type=int, default=100)
    args = ap.parse_args()

    print("1) 解包 tar ...")
    for mod in ("T1", "T2"):
        tar = os.path.join(DATA, f"IXI-{mod}-first800.tar")
        if not os.path.exists(tar):
            print(f"  缺少 {tar}，跳过"); return
        untar_partial(tar, os.path.join(RAW, mod))

    t1_files = {subject_id(f): os.path.join(RAW, "T1", f)
                for f in os.listdir(os.path.join(RAW, "T1"))}
    t2_files = {subject_id(f): os.path.join(RAW, "T2", f)
                for f in os.listdir(os.path.join(RAW, "T2"))}
    pairs = sorted(set(t1_files) & set(t2_files))[: args.max_sub]
    print(f"2) 配对受试者: {len(pairs)} 个（如 {pairs[:3]}）")

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(EX, exist_ok=True)
    rows = ["subject,split,slice,t1_path,t2_path"]
    methods = {}
    n_sub = 0
    demo_saved = 0
    for si, sid in enumerate(pairs):
        split = "val" if si < args.n_val else "demo"
        try:
            t1, t2, method = load_pair(t1_files[sid], t2_files[sid])
        except Exception as e:
            print(f"  跳过 {sid}: {e}")
            continue
        methods[method] = methods.get(method, 0) + 1
        if t1.shape[:2] != (256, 256) or t2.shape[:2] != (256, 256):
            continue
        t1n, t2n = norm01(t1), norm01(t2)
        sel = pick_slices(t1n, t2n, args.slices_per_sub)
        for k in sel:
            a, b = t1n[:, :, k].astype(np.float32), t2n[:, :, k].astype(np.float32)
            pa = os.path.join(OUT, f"{sid}_z{k:03d}_T1.npy")
            pb = os.path.join(OUT, f"{sid}_z{k:03d}_T2.npy")
            np.save(pa, a); np.save(pb, b)
            rows.append(f"{sid},{split},{k},{pa},{pb}")
            # 每个受试者存 1 张中间切片 PNG 作演示输入
            if demo_saved < 12 and k == sel[len(sel) // 2]:
                from PIL import Image
                src_mod, src = ("T2", b) if si % 2 == 0 else ("T1", a)
                Image.fromarray((src * 255).astype(np.uint8)).save(
                    os.path.join(EX, f"demo_{sid}_{src_mod}.png"))
                demo_saved += 1
        n_sub += 1
        print(f"  [{n_sub}/{len(pairs)}] {sid} method={method} slices={len(sel)}")
    with open(os.path.join(OUT, "manifest.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    print(f"完成: {n_sub} 受试者, {len(rows)-1} 切片, 对齐方法 {methods}, 清单 -> {OUT}/manifest.csv")


if __name__ == "__main__":
    main()
