# -*- coding: utf-8 -*-
"""Gate 3 · 离线派生（从 maps_full npz）：三层 AUROC / AUPRC / K 消融 /
CReF–U 相关 / AUSE / risk-coverage / 体区十分箱 + ΔMAE。

用法:
  python scripts/eval_derive.py --group ixi_test_t1t2
  python scripts/eval_derive.py --group synthrad_t1ct --w 0.6
输出: results/maps_full/<group>/derived.json
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

K_GRID = [1, 2, 3, 5, 10]
W_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def pct_norm(m, q=99.0):
    hi = np.percentile(m, q)
    if hi <= 1e-12:
        return np.zeros_like(m)
    return np.clip(m / hi, 0.0, 1.0)


def auroc_auprc(score, label):
    """score 高=可疑；label bool。返回 (auroc, auprc)。"""
    order = np.argsort(score)
    lab = label[order].astype(int)
    npos = int(lab.sum()); nneg = len(lab) - npos
    if npos < 10 or nneg < 10:
        return float("nan"), float("nan")
    ranks = np.empty(len(lab), np.float64)
    ranks[np.argsort(score, kind="mergesort")] = np.arange(1, len(lab) + 1)
    au = (ranks[label == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    # AUPRC（按分数降序阶梯积分）
    lab_desc = label[np.argsort(-score)].astype(int)
    tp = np.cumsum(lab_desc)
    prec = tp / (np.arange(len(lab_desc)) + 1)
    rec = tp / npos
    auprc = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
    return float(au), auprc


def risk_coverage(trust, err):
    order = np.argsort(-trust)
    er = err[order]
    n = max(len(er), 1)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(er) / np.arange(1, n + 1)
    return cov, risk


def ause(trust, err):
    cov, risk = risk_coverage(trust, err)
    cov_o, risk_o = risk_coverage(-err, err)      # oracle：误差升序保留
    grid = np.linspace(0, 1, 100)
    return float(np.trapezoid(np.interp(grid, cov, risk) -
                              np.interp(grid, cov_o, risk_o), grid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--w", type=float, default=0.6, help="融合权重 w1（CReF）")
    args = ap.parse_args()

    run_dir = os.path.join(ROOT, "results", "maps_full", args.group)
    npzs = sorted(glob.glob(os.path.join(run_dir, "*.npz")))
    if not npzs:
        print("no npz in", run_dir); return

    per_slice = []
    fps = set()
    for p in npzs:
        try:
            d = np.load(p, allow_pickle=True)
        except Exception:
            continue
        fps.add(str(d["H_config"]))
        pred = d["pred"].astype(np.float32)
        gt = d["gt"].astype(np.float32)
        mask = d["mask"]
        if mask.sum() < 200:
            continue
        rcrf = d["rcrf_raw"].astype(np.float32)
        unc = d["unc_raw"].astype(np.float32)
        steps = d["step_sums"].astype(np.float32)   # [10,H,W]（未捕获步=0）
        ts = d["ts"].astype(int)
        w = float(args.w)

        g, pr = gt[mask], pred[mask]
        gn = g / max(g.max(), 1e-9)
        pn = pr / max(pr.max(), 1e-9) if pr.max() > 1e-9 else pr * 0
        err = np.abs(gn - pn)

        # K 消融（捕获窗）：K=取 ts 中最大的 K 个捕获步（最靠近数据端）
        row = {"id": str(d["id"]), "z": int(d["z"]),
               "body_psnr": float(10 * np.log10(
                   max(gn.max(), 1e-9) ** 2 / max(np.mean((gn - pn) ** 2), 1e-12)))}
        for K in K_GRID:
            keep_ts = set(sorted(ts, reverse=True)[:K])
            sel = np.isin(ts, list(keep_ts))
            rcrf_k = (steps[sel].sum(axis=0) / (K * 1.0)).astype(np.float32)
            # 链维已并入 step_sums（链间求和），分母 K·N：N=4
            rcrf_k = steps[sel].sum(axis=0) / (K * 4.0)
            u = unc[mask]
            r_n = pct_norm(rcrf_k[mask]); u_n = pct_norm(u)
            trust = np.clip(1 - (args.w * r_n + (1 - args.w) * u_n), 0, 1)
            hi_e = np.percentile(err, 90)
            lab = err >= hi_e
            au, ap_ = auroc_auprc(1 - trust, lab)
            row[f"auroc_K{K}"] = round(au, 4)
            if K == 3:
                row["auprc"] = round(ap_, 4)
                row["pos_rate"] = round(float(lab.mean()), 4)
                row["cref_unc_spearman"] = round(float(spearmanr(
                    rcrf_k[mask], u).statistic), 4)
                # 风险覆盖 / AUSE（w* 口径）
                row["ause"] = round(ause(trust, err), 5)
                tr_lo = err[trust <= np.percentile(trust, 10)].mean()
                tr_hi = err[trust >= np.percentile(trust, 90)].mean()
                row["decile_mae_low"] = round(float(tr_lo), 5)
                row["decile_mae_high"] = round(float(tr_hi), 5)
                row["decile_mae_diff"] = round(float(tr_lo - tr_hi), 5)
                row["decile_sep"] = round(float(tr_lo / max(tr_hi, 1e-9)), 2)
        per_slice.append(row)

    ids = sorted({r["id"] for r in per_slice})
    out = {"group": args.group, "w": args.w, "H_configs": sorted(fps),
           "n_slices": len(per_slice), "n_subjects": len(ids), "subjects": ids,
           "per_slice": per_slice}

    def agg(key):
        v = [r[key] for r in per_slice if key in r and np.isfinite(r[key])]
        return {"mean": round(float(np.mean(v)), 4),
                "std": round(float(np.std(v)), 4)} if v else None

    for K in K_GRID:
        out[f"slice_macro_auroc_K{K}"] = agg(f"auroc_K{K}")
    out["slice_macro_auprc"] = agg("auprc")
    out["slice_macro_auroc_w{:.1f}".format(args.w)] = agg("auroc_K3")
    out["ause"] = agg("ause")
    out["cref_unc_spearman"] = agg("cref_unc_spearman")
    out["decile"] = {k: agg(k) for k in
                     ("decile_mae_low", "decile_mae_high", "decile_mae_diff", "decile_sep")}

    with open(os.path.join(run_dir, "derived.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    slim = {k: v for k, v in out.items() if k != "per_slice"}
    print(json.dumps(slim, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
