# -*- coding: utf-8 -*-
"""融合权重扫描（体区口径，maps_full npz 离线复算）。

w = w_CReF ∈ {0,0.2,...,1.0}，trust = 1-(w*pct_norm(rcrf_K3)+(1-w)*pct_norm(unc))，
within-body AUROC（top-10% 误差为正类），逐片宏平均。与 eval_derive 同口径，
只是扫 w 而非 K。等预算：全部用 N=4 链求和残差（40 前传预算不变）。
输出: results/<date>_weight_scan/summary.json
"""
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GROUPS = ["ixi_test_t1t2", "ixi_test_t2t1", "synthrad_t1ct", "synthrad_t2ct"]
W_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
K0, N_ENS = 3, 4


def pct_norm(v):
    hi = np.percentile(v, 99)
    if hi <= 1e-12:
        return np.zeros_like(v)
    return np.clip(v / hi, 0.0, 1.0)


def auroc(score, lab):
    pos, neg = score[lab], score[~lab]
    npos, nneg = len(pos), len(neg)
    if npos < 10 or nneg < 10:
        return np.nan
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(order), np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[lab].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "protocol":
           "within-body AUROC, maps_full npz, K0=3 data-end, N=4 chain-summed CReF, pct99 norm",
           "w_grid": W_GRID, "groups": {}}
    for g in GROUPS:
        files = sorted(glob.glob(os.path.join(
            ROOT, "results", "maps_full", g, "*.npz")))
        per = {w: [] for w in W_GRID}
        for f in files:
            d = np.load(f, allow_pickle=True)
            mask = d["mask"]
            if mask.sum() < 200:
                continue
            gt, pred = d["gt"].astype(np.float32), d["pred"].astype(np.float32)
            g_, pr = gt[mask], pred[mask]
            gn = g_ / max(g_.max(), 1e-9)
            pn = pr / max(pr.max(), 1e-9) if pr.max() > 1e-9 else pr * 0
            err = np.abs(gn - pn)
            steps = d["step_sums"].astype(np.float32)
            ts = d["ts"].astype(int)
            keep = sorted(ts)[:K0]
            rcrf = steps[np.isin(ts, keep)].sum(axis=0) / (K0 * float(N_ENS))
            unc = d["unc_raw"].astype(np.float32)
            r_n, u_n = pct_norm(rcrf[mask]), pct_norm(unc[mask])
            lab = err >= np.percentile(err, 90)
            for w in W_GRID:
                trust = np.clip(1 - (w * r_n + (1 - w) * u_n), 0, 1)
                per[w].append(auroc(1 - trust, lab))
        agg = {}
        for w in W_GRID:
            v = np.array(per[w], dtype=np.float64)
            v = v[~np.isnan(v)]
            agg[f"w{w:.1f}"] = {"mean": round(float(v.mean()), 4),
                                "std": round(float(v.std(ddof=1)), 4),
                                "n": int(v.size)}
        out["groups"][g] = agg
        print(g, {k: v["mean"] for k, v in agg.items()}, flush=True)
    run_dir = os.path.join(ROOT, "results",
                           f"{time.strftime('%Y-%m-%d')}_weight_scan")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("->", run_dir)


if __name__ == "__main__":
    main()
