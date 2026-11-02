"""UQ 基线对照（体区口径）：我们的信任信号 vs 等预算基线。

背景：消融表（docs/08）的 AUROC/sparsification 在整图上计算，背景空气
（高信任、低误差）会虚高指标——肿瘤分析已证实该效应（0.871→0.51）。
本脚本把所有方法统一到体掩码（src>0.02）内重新计算，并补两个推理端基线：

  - TTA dispersion   ：4 个几何增广（原/上下翻/左右翻/旋转90+逆变换）各推理 1 链，
                       对逆变换回原坐标的预测取逐像素标准差（预算 4×10=40 前传）
  - input-noise disp ：输入加 σ=0.02 高斯噪声 ×4 次独立推理，取标准差（预算 40）

对照我们的（同预算）：fused(CReF+集成 w.6/.4)、unc_n4、rcrf_n4、rcrf_n1（预算 10），
外加梯度基线与随机对照。

用法: python scripts/eval_uq_baselines.py --task T1->T2 [--n 60]
输出: results/<date>_uq_baselines_<task>/{per_slice.csv,summary.json}
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

from trustbridge.engine import BridgeEngine, TASKS  # noqa: E402
from trustbridge.pipeline import translate_single  # noqa: E402
from trustbridge.metrics import mean_norm  # noqa: E402
from trustbridge.sampler import _pct_norm  # noqa: E402

NOISE_SIGMA = 0.02


def auroc_low_vs_higherr(trust: np.ndarray, err: np.ndarray) -> float:
    """低信任预测体区内高误差（>90 分位）的 AUROC。"""
    hi = err > np.percentile(err, 90)
    npos, nneg = int(hi.sum()), int((~hi).sum())
    if npos < 10 or nneg < 10:
        return float("nan")
    score = (1.0 - trust).ravel()
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    r_sum = ranks[hi.ravel()].sum()
    return (r_sum - npos * (npos + 1) / 2) / (npos * nneg)


def masked_psnr(gt, pred, mask):
    g, p = mean_norm(gt), mean_norm(pred)
    return float(peak_signal_noise_ratio(g[mask], p[mask], data_range=g[mask].max()))


def sparsify_gain(gt, pred, trust, mask, drop_pct):
    """体区内剔除信任最低 drop_pct% 像素后的 PSNR 提升。"""
    g, p = mean_norm(gt), mean_norm(pred)
    p_full = peak_signal_noise_ratio(g[mask], p[mask], data_range=g[mask].max())
    tr = trust[mask]
    keep = tr >= np.percentile(tr, drop_pct)
    if keep.sum() < 100:
        return 0.0
    return float(peak_signal_noise_ratio(g[mask][keep], p[mask][keep],
                                         data_range=g[mask].max()) - p_full)


def decile_separation(trust, err, mask):
    """体区内最不可信十分位与最可信十分位的平均绝对误差比。"""
    tr, er = trust[mask], err[mask]
    lo = er[tr <= np.percentile(tr, 10)].mean()
    hi = er[tr >= np.percentile(tr, 90)].mean()
    return float(lo / max(hi, 1e-9))


def eval_predictor(trust, gt, pred, err, mask):
    rho = float(spearmanr(trust[mask], err[mask]).statistic)
    return {"spearman": round(rho, 4),
            "auroc": round(auroc_low_vs_higherr(trust, err), 4),
            "dpsnr10": round(sparsify_gain(gt, pred, trust, mask, 10), 3),
            "dpsnr20": round(sparsify_gain(gt, pred, trust, mask, 20), 3),
            "dpsnr30": round(sparsify_gain(gt, pred, trust, mask, 30), 3),
            "decile_sep": round(decile_separation(trust, err, mask), 1)}


TTA_VARIANTS = [("id", lambda a: a, lambda a: a),
                ("flipud", np.flipud, np.flipud),
                ("fliplr", np.fliplr, np.fliplr),
                ("rot90", lambda a: np.rot90(a, 1), lambda a: np.rot90(a, -1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T1->T2")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_v3.csv"), encoding="utf-8")))
    rows = [r for r in man if r["split"] == "test"]
    step = max(1, len(rows) // args.n)
    rows = rows[::step][: args.n]

    eng = BridgeEngine(args.task)   # 默认自训练 ckpt（与消融一致）
    _, src_mod, tgt_mod, _ = TASKS[args.task]
    run_dir = os.path.join(ROOT, "results",
                           f"{time.strftime('%Y-%m-%d')}_uq_baselines_{args.task.replace('->', '_')}")
    os.makedirs(run_dir, exist_ok=True)

    per, agg = [], {}

    def record(cfg, r, pred, trust, err, mask, gt, fwd, elapsed, psnr=None):
        m = eval_predictor(trust, gt, pred, err, mask)
        m["fwd"] = fwd
        m["elapsed_s"] = round(elapsed, 2)
        if psnr is not None:
            m["body_psnr"] = round(psnr, 2)
        per.append({"task": args.task, "subject": r["subject"], "z": r["z"],
                    "config": cfg, **m})
        for k, v in m.items():
            agg.setdefault((cfg, k), []).append(v)

    t00 = time.time()
    for i, r in enumerate(rows):
        t1 = np.load(r["t1_path"]).astype(np.float32)
        t2 = np.load(r["t2_path"]).astype(np.float32)
        src, gt = (t2, t1) if src_mod == "T2" else (t1, t2)
        mask = src > 0.02
        err_needed = None

        # ---- 我们的：N=4 融合（同一次推理拆出 rcrf/unc）----
        out = translate_single(eng, src, n_ensemble=4, step_skip=1, seed=args.seed)
        res, pred = out["result"], out["pred01"]
        err = np.abs(gt - pred); err_needed = err
        fwd = sum(sum(c) for c in res.recursions_used)
        R, U = _pct_norm(res.rcrf_map), _pct_norm(res.unc_map)
        trust = 1 - (0.6 * R + 0.4 * U)
        record("ours_fused", r, pred, trust, err, mask, gt, fwd, res.elapsed_s,
               masked_psnr(gt, pred, mask))
        record("ours_unc_n4", r, pred, 1 - U, err, mask, gt, fwd, 0.0)
        record("ours_rcrf_n4", r, pred, 1 - R, err, mask, gt, fwd, 0.0)

        # ---- CReF 单链（预算 10）----
        out1 = translate_single(eng, src, n_ensemble=1, step_skip=1, seed=args.seed)
        res1, pred1 = out1["result"], out1["pred01"]
        err1 = np.abs(gt - pred1)
        record("ours_rcrf_n1", r, pred1, 1 - _pct_norm(res1.rcrf_map), err1, mask,
               gt, sum(sum(c) for c in res1.recursions_used), res1.elapsed_s,
               masked_psnr(gt, pred1, mask))

        # ---- 基线①：TTA dispersion（4 增广 × N=1，预算 40）----
        t0 = time.time()
        preds = []
        for name, fwd_t, inv_t in TTA_VARIANTS:
            o = translate_single(eng, fwd_t(np.ascontiguousarray(src)),
                                 n_ensemble=1, step_skip=1, seed=args.seed)
            p = inv_t(np.ascontiguousarray(o["pred01"]))
            if p.shape != src.shape:
                p = p[:src.shape[0], :src.shape[1]]
            preds.append(p)
        tta_std = np.std(preds, axis=0)
        record("tta_disp", r, pred, 1 - _pct_norm(tta_std), err, mask, gt,
               4 * 10, time.time() - t0)

        # ---- 基线②：input-noise dispersion（4 噪声 × N=1，预算 40）----
        t0 = time.time()
        preds = []
        for k in range(4):
            rng = np.random.RandomState(args.seed + 100 + k)
            noisy = np.clip(src + rng.normal(0, NOISE_SIGMA, src.shape), 0, 1)
            o = translate_single(eng, noisy.astype(np.float32),
                                 n_ensemble=1, step_skip=1, seed=args.seed)
            preds.append(o["pred01"])
        noise_std = np.std(preds, axis=0)
        record("noise_disp", r, pred, 1 - _pct_norm(noise_std), err, mask, gt,
               4 * 10, time.time() - t0)

        # ---- 梯度基线 / 随机对照 ----
        gy, gx = np.gradient(src)
        grad = np.hypot(gx, gy)
        record("gradient", r, pred, _pct_norm(grad), err, mask, gt, 0, 0.0)
        rng = np.random.RandomState(args.seed + i)
        record("random", r, pred, rng.rand(*err.shape), err, mask, gt, 0, 0.0)

        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            print(f"  [{i+1}/{len(rows)}] ours {np.mean(agg[('ours_fused','auroc')]):.3f} | "
                  f"tta {np.mean(agg[('tta_disp','auroc')]):.3f} | "
                  f"noise {np.mean(agg[('noise_disp','auroc')]):.3f} | "
                  f"总耗时 {(time.time()-t00)/60:.1f}min", flush=True)

    # ---- 汇总 ----
    summary = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "task": args.task,
               "n_slices": len(rows), "seed": args.seed,
               "protocol": "within-body (src>0.02); TTA=4几何增广xN1; noise=σ0.02x4xN1; 等预算40前传",
               "methods": {}}
    keys = ["spearman", "auroc", "dpsnr20", "dpsnr30", "decile_sep"]
    for cfg in ("ours_fused", "ours_unc_n4", "ours_rcrf_n4", "ours_rcrf_n1",
                "tta_disp", "noise_disp", "gradient", "random"):
        summary["methods"][cfg] = {
            k: {"mean": round(float(np.nanmean(agg[(cfg, k)])), 4),
                "std": round(float(np.nanstd(agg[(cfg, k)])), 4)} for k in keys}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "per_slice.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader(); w.writerows(per)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("->", run_dir)


if __name__ == "__main__":
    main()
