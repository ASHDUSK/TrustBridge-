"""测试集评估：官方口径 PSNR/SSIM + NCC + 误差分解（线性强度拟合）。

用法:
  python scripts/eval_test.py --task T2->T1 --ckpt checkpoints/self_t2t1.ckpt --n 120
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.stats import pearsonr, spearmanr

from trustbridge.engine import BridgeEngine, TASKS
from trustbridge.pipeline import translate_single
from trustbridge.metrics import mean_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T2->T1")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_v3.csv"), encoding="utf-8")))
    rows = [r for r in man if r["split"] == "test"]
    step = max(1, len(rows) // args.n)
    rows = rows[::step][: args.n]
    display, src_mod, tgt_mod, _ckpt = TASKS[args.task]
    eng = BridgeEngine(args.task, ckpt_path=args.ckpt)
    print(f"评估 {len(rows)} 切片 | task={args.task} | ckpt={args.ckpt or '官方'}")

    ps, ss, nccs, fits, offs = [], [], [], [], []
    for i, r in enumerate(rows):
        src = np.load(r["t2_path"] if src_mod == "T2" else r["t1_path"]).astype(np.float32)
        gt = np.load(r["t1_path"] if tgt_mod == "T1" else r["t2_path"]).astype(np.float32)
        out = translate_single(eng, src, n_ensemble=1, step_skip=1, seed=args.seed)
        pred = out["pred01"]
        g, p = mean_norm(gt), mean_norm(pred)
        ps.append(peak_signal_noise_ratio(g, p, data_range=g.max()))
        ss.append(structural_similarity(g, p, data_range=g.max()) * 100)
        a = g - g.mean(); b = p - p.mean()
        nccs.append(float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9)))
        # 线性强度拟合（尺度已被 mean_norm 消除，看残余线性畸变）
        A = np.stack([g.ravel(), np.ones(g.size)], 1)
        (ka, kb), *_ = np.linalg.lstsq(A, p.ravel(), rcond=None)
        offs.append(kb)
        fits.append(peak_signal_noise_ratio(g, ka * g + kb, data_range=g.max()))
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(rows)}] PSNR {np.mean(ps):.2f} SSIM {np.mean(ss):.2f}% NCC {np.mean(nccs):.3f}")

    res = {
        "task": args.task, "ckpt": args.ckpt or "official", "n_slices": len(rows),
        "psnr": round(float(np.mean(ps)), 2), "psnr_std": round(float(np.std(ps)), 2),
        "ssim": round(float(np.mean(ss)), 2), "ssim_std": round(float(np.std(ss)), 2),
        "ncc": round(float(np.mean(nccs)), 4),
        "psnr_after_linear_fit": round(float(np.mean(fits)), 2),
        "mean_offset_term": round(float(np.mean(offs)), 4),
    }
    print(json.dumps(res, ensure_ascii=False, indent=2))
    out = args.out or os.path.join(ROOT, "outputs", f"eval_{args.task.replace('->','_').replace('*','_self')}_{os.path.basename(args.ckpt or 'official')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("->", out)


if __name__ == "__main__":
    main()
