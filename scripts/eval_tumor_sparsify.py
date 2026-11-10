"""BraTS2018 体区口径复检：sparsification 增益 + 校准分离（修正空气像素效应）。

背景：+6.95 dB（剔除最低信任 20% 像素的 PSNR 增益）在整图口径下计算，
与被撤回的 0.871 同源——背景空气高信任低误差，会虚高体区内的表观增益。
本脚本在体掩码（FLAIR>0.02）内重算：
  - dpsnr10/20/30（体区 sparsification 增益，fused 信任图）
  - 低信任→高误差 AUROC（体区内，top-decile 误差为正例）
  - 十分位校准分离（体区内最不可信/最可信十分位平均绝对误差比）
  - 体区 PSNR（切片级，供 QC 阈值分析）

用法: python scripts/eval_tumor_sparsify.py [--n 60]
输出: results/<date>_tumor_sparsify_body/
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr
from skimage.metrics import peak_signal_noise_ratio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trustbridge.engine import BridgeEngine  # noqa: E402
from trustbridge.pipeline import translate_single  # noqa: E402
from trustbridge.metrics import mean_norm  # noqa: E402


def auroc_low_vs_higherr(trust, err):
    hi = err > np.percentile(err, 90)
    npos, nneg = int(hi.sum()), int((~hi).sum())
    if npos < 10 or nneg < 10:
        return float("nan")
    score = (1.0 - trust).ravel()
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    return float((ranks[hi.ravel()].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "brats2018_slices",
                                                "manifest.csv"), encoding="utf-8")))
    man_t = [r for r in man if float(r["tumor_frac"]) > 0.005]
    step = max(1, len(man_t) // args.n)
    sel = man_t[::step][: args.n]
    print(f"评估 {len(sel)} 个含瘤切片（tumor_frac>0.005，体区口径）", flush=True)

    eng = BridgeEngine("BRATS_FLAIR->T2",
                       ckpt_path=os.path.join(ROOT, "checkpoints", "brats_flair_t2.ckpt"))
    run_dir = os.path.join(ROOT, "results", f"{time.strftime('%Y-%m-%d')}_tumor_sparsify_body")
    os.makedirs(run_dir, exist_ok=True)

    per = []
    t00 = time.time()
    for i, r in enumerate(sel):
        flair = np.load(r["flair_path"]).astype(np.float32)
        t2gt = np.load(r["t2_path"]).astype(np.float32)
        body = flair > 0.02

        out = translate_single(eng, flair, n_ensemble=4, step_skip=1, seed=args.seed)
        res, pred = out["result"], out["pred01"]
        R_n = _pct_norm(res.rcrf_raw_map)
        U_n = _pct_norm(res.unc_raw_map)
        trust = np.clip(1 - 0.6 * R_n - 0.4 * U_n, 0, 1)
        if trust.shape != flair.shape:
            S = trust.shape[0]; H, W = flair.shape
            pt, pl = (S - H) // 2, (S - W) // 2
            crop = lambda m: np.ascontiguousarray(m[pt:pt + H, pl:pl + W])
            trust, pred = crop(trust), crop(pred)
        err = np.abs(t2gt - pred)

        g, p = mean_norm(t2gt), mean_norm(pred)
        psnr_body = float(peak_signal_noise_ratio(g[body], p[body], data_range=g[body].max()))

        def sparsify(drop_pct):
            tr = trust[body]
            keep = tr >= np.percentile(tr, drop_pct)
            if keep.sum() < 100:
                return 0.0
            return float(peak_signal_noise_ratio(g[body][keep], p[body][keep],
                                                 data_range=g[body].max()) - psnr_body)

        tr_b, er_b = trust[body], err[body]
        lo = er_b[tr_b <= np.percentile(tr_b, 10)].mean()
        hi = er_b[tr_b >= np.percentile(tr_b, 90)].mean()
        per.append({"patient": r["patient"], "z": r["z"],
                    "auroc": round(auroc_low_vs_higherr(trust, err), 4),
                    "spearman": round(float(spearmanr(tr_b, er_b).statistic), 4),
                    "dpsnr10": round(sparsify(10), 3),
                    "dpsnr20": round(sparsify(20), 3),
                    "dpsnr30": round(sparsify(30), 3),
                    "decile_sep": round(float(lo / max(hi, 1e-9)), 1),
                    "body_psnr": round(psnr_body, 2),
                    "trust_score": round(float(tr_b.mean() * 100), 1)})
        if (i + 1) % 10 == 0 or i == len(sel) - 1:
            print(f"  [{i+1}/{len(sel)}] AUROC {np.nanmean([q['auroc'] for q in per]):.3f} | "
                  f"dpsnr20 {np.mean([q['dpsnr20'] for q in per]):.2f} | "
                  f"{(time.time()-t00)/60:.1f}min", flush=True)

    agg = {k: {"mean": round(float(np.nanmean([q[k] for q in per])), 4),
               "std": round(float(np.nanstd([q[k] for q in per])), 4)}
           for k in ("auroc", "spearman", "dpsnr10", "dpsnr20", "dpsnr30",
                     "decile_sep", "body_psnr", "trust_score")}
    summary = {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
               "n_slices": len(per), "seed": args.seed,
               "protocol": "within-body (FLAIR>0.02)，official FLAIR->T2 256² 权重，N=4",
               "note": "体区口径复检：旧 +6.95dB 为整图口径，与被撤回的 0.871 同源。",
               **agg}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "per_slice.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader(); w.writerows(per)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("->", run_dir)


if __name__ == "__main__":
    main()
