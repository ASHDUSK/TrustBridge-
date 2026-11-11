# -*- coding: utf-8 -*-
"""体掩码阈值敏感性：mask = source > thr, thr ∈ {0.01, 0.02, 0.05}。

兑现论文 §Setup "threshold sensitivity at 0.01/0.05" 的承诺（纯 CPU 离线复算）。
对 maps_full 已存 npz 重算：在每种阈值掩码内重做掩码内归一化、K=3 CReF、
0.6/0.4 融合与 top-10% 误差 AUROC/AUPRC。thr=0.02 应复现 derived.json（自校验）。

用法: python scripts/eval_mask_threshold_sensitivity.py
输出: results/<date>_mask_threshold_sensitivity/summary.json
"""
import csv
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

GROUPS = ["ixi_test_t1t2", "ixi_test_t2t1", "synthrad_t1ct", "synthrad_t2ct"]
THRESHOLDS = [0.01, 0.02, 0.05]
W_CREF = 0.6
K0, N_ENS = 3, 4


def pct_norm(v):
    q = np.percentile(v, 99)
    return np.clip(v / max(q, 1e-9), 0, 1) if q > 1e-9 else v * 0


def auroc(score, lab):
    pos, neg = score[lab], score[~lab]
    if len(pos) < 10 or len(neg) < 10:
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_manifest_index():
    idx = {}
    ixi = list(csv.DictReader(open(os.path.join(
        ROOT, "data", "ixi_slices", "manifest_v3.csv"), encoding="utf-8")))
    for r in ixi:
        idx[("ixi", r["subject"], int(r["z"]))] = r
    for name, key in [("ct_a", "t1ct"), ("ct_c", "t2ct")]:
        rows = list(csv.DictReader(open(os.path.join(
            ROOT, "data", "synthrad_slices_128", f"manifest_{name}.csv"),
            encoding="utf-8")))
        for r in rows:
            idx[("synthrad", r["patient"], int(r["z"]), key)] = r
    return idx


def source_path(group, sid, z, midx):
    if group.startswith("ixi"):
        r = midx[("ixi", str(sid), int(z))]
        return r["t1_path"] if group.endswith("t1t2") else r["t2_path"]
    key = "t1ct" if group.endswith("t1ct") else "t2ct"
    r = midx[("synthrad", str(sid), int(z), key)]
    return r["mr_path"]


def main():
    from scipy.stats import spearmanr  # noqa: F401  （保持与 derive 相同依赖面）
    midx = load_manifest_index()
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "w": W_CREF,
           "K0": K0, "thresholds": THRESHOLDS, "groups": {}}
    for group in GROUPS:
        files = sorted(glob.glob(os.path.join(
            ROOT, "results", "maps_full", group, "*.npz")))
        rows = []
        for f in files:
            d = np.load(f, allow_pickle=True)
            sid, z = str(d["id"]), int(d["z"])
            mask02 = d["mask"]
            src = np.load(os.path.join(ROOT, source_path(group, sid, z, midx))
                          ).astype(np.float32)
            if not np.array_equal(src > 0.02, mask02):
                print(f"  [warn] mask mismatch {group}/{sid}_z{z}", flush=True)
            gt = d["gt"].astype(np.float32)
            pred = d["pred"].astype(np.float32)
            steps = d["step_sums"].astype(np.float32)
            ts = d["ts"].astype(int)
            keep = set(sorted(ts)[:K0])
            rcrf_k = steps[np.isin(ts, list(keep))].sum(axis=0) / (K0 * float(N_ENS))
            unc = d["unc_raw"].astype(np.float32)
            row = {"id": sid, "z": z}
            for thr in THRESHOLDS:
                m = src > thr
                if m.sum() < 200:
                    row[f"auroc_{thr}"] = np.nan
                    continue
                g, pr = gt[m], pred[m]
                gn = g / max(g.max(), 1e-9)
                pn = pr / max(pr.max(), 1e-9) if pr.max() > 1e-9 else pr * 0
                err = np.abs(gn - pn)
                r_n = pct_norm(rcrf_k[m])
                u_n = pct_norm(unc[m])
                trust = np.clip(1 - (W_CREF * r_n + (1 - W_CREF) * u_n), 0, 1)
                lab = err >= np.percentile(err, 90)
                row[f"auroc_{thr}"] = round(auroc(1 - trust, lab), 4)
            rows.append(row)
            if len(rows) % 50 == 0:
                print(f"  [{group}] {len(rows)}/{len(files)}", flush=True)
        agg = {}
        for thr in THRESHOLDS:
            v = np.array([r[f"auroc_{thr}"] for r in rows], dtype=np.float64)
            v = v[~np.isnan(v)]
            agg[f"auroc_thr{thr}"] = {
                "mean": round(float(v.mean()), 4),
                "sd": round(float(v.std(ddof=1)), 4), "n": int(v.size)}
        out["groups"][group] = {"n_slices": len(rows), "agg": agg}
        print(f"{group}: " + " | ".join(
            f"thr{t}: {agg[f'auroc_thr{t}']['mean']}" for t in THRESHOLDS),
            flush=True)
    run_dir = os.path.join(ROOT, "results",
                           f"{time.strftime('%Y-%m-%d')}_mask_threshold_sensitivity")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("->", run_dir)


if __name__ == "__main__":
    main()
