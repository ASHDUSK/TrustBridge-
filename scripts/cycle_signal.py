"""往返一致性第三信号（Cycle-Consistency Residual, CCR）。

原理：T2 →(模型A)→ T1' →(模型B)→ T2''，回译结果 T2'' 与原始 T2 的逐像素差异
反映"信息在翻译链中丢失/被编造的位置"。与既有两信号（收敛稳定性、采样一致性）
正交：它度量的是【往返可逆性】，而前两者度量的是【单次翻译的内部稳定性】。

验证问题：CCR 是否与真实误差 |x̂−GT| 相关？三信号融合能否超越双信号？
协议：双方向（T2→T1 主任务用 A→B 回译；T1→T2 主任务用 B→A 回译），
      val 调融合权重、test 报告，与 fixed 双信号基线同片对照。
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scipy.stats import spearmanr  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio  # noqa: E402

from trustbridge.engine import BridgeEngine, TASKS  # noqa: E402
from trustbridge.metrics import mean_norm  # noqa: E402
from trustbridge.pipeline import translate_single  # noqa: E402
from trustbridge.sampler import _pct_norm  # noqa: E402


def auroc_score(score, pos):
    s = score.ravel().astype(np.float64); p = pos.ravel()
    npos, nneg = int(p.sum()), s.size - int(p.sum())
    if npos == 0 or nneg == 0:
        return np.nan
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1)
    return float((ranks[p].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def drop_gain_px(g, p, trust, pct=20):
    k = int(pct / 100 * trust.size)
    drop = np.zeros(trust.size, dtype=bool)
    drop[np.argsort(trust.ravel())[:k]] = True     # 信任最低 = 剔除
    keep = (~drop).reshape(g.shape)
    if keep.sum() < 100:
        return np.nan
    return float(peak_signal_noise_ratio(g[keep], p[keep], data_range=g.max()) -
                 peak_signal_noise_ratio(g, p, data_range=g.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T2->T1")          # 主任务方向
    ap.add_argument("--ckpt-main", default=None)          # 主任务（正向）权重
    ap.add_argument("--ckpt-back", default=None)          # 回译方向权重
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-ens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    # 主任务 T2->T1：正向 A=self_t2t1，回译 B=self_t1t2
    back_task = "T1->T2" if args.task == "T2->T1" else "T2->T1"
    ckpt_main = args.ckpt_main or os.path.join(
        ROOT, "checkpoints", TASKS[args.task][3].replace("ixi_", "self_")
        if not TASKS[args.task][3].startswith("self_") else TASKS[args.task][3])
    ckpt_back = args.ckpt_back or os.path.join(
        ROOT, "checkpoints", TASKS[back_task][3].replace("ixi_", "self_")
        if not TASKS[back_task][3].startswith("self_") else TASKS[back_task][3])

    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_v3.csv"), encoding="utf-8")))
    for split in ("val", "test"):
        pass
    runs = {}
    for split in ("val", "test"):
        rows = [r for r in man if r["split"] == split]
        step = max(1, len(rows) // args.n)
        runs[split] = rows[::step][: args.n]

    eng_main = BridgeEngine(args.task, ckpt_path=ckpt_main)
    eng_back = BridgeEngine(back_task, ckpt_path=ckpt_back)
    _, src_mod, tgt_mod, _ = TASKS[args.task]

    summary = {"task": args.task, "date": datetime.now().isoformat(timespec="seconds"),
               "n": {s: len(r) for s, r in runs.items()},
               "schemes": {}}
    per_all = []
    for split, rows in runs.items():
        schemes = {"fixed2": [], "ccr_only": [], "fusion3_eq": [], "fusion3_val": [],
                   "oracle": []}
        w3 = None   # val 上调好的三信号权重
        for i, r in enumerate(rows):
            src = np.load(r["t2_path"] if src_mod == "T2" else r["t1_path"]).astype(np.float32)
            gt = np.load(r["t1_path"] if tgt_mod == "T1" else r["t2_path"]).astype(np.float32)
            # 正向（含信任信号）
            out = translate_single(eng_main, src, n_ensemble=args.n_ens,
                                   step_skip=1, seed=args.seed)
            res, pred = out["result"], out["pred01"]
            # 回译：pred → src 模态
            out_b = translate_single(eng_back, pred, n_ensemble=1, step_skip=1,
                                     seed=args.seed)
            ccr = norm01 if False else _pct_norm(np.abs(out_b["pred01"] - src))
            err = np.abs(gt - pred)
            g, p = mean_norm(gt), mean_norm(pred)

            r_hat, u_hat = norm01(res.rcrf_map), norm01(res.unc_map)
            fixed2 = 0.6 * r_hat + 0.4 * u_hat
            ccr_n = _pct_norm(ccr)
            if split == "val":
                best = None
                for w in np.arange(0, 1.01, 0.1):
                    sus = (1 - w) * fixed2 + w * ccr_n
                    rho = float(spearmanr(sus.ravel(), err.ravel()).statistic)
                    if best is None or rho > best[0]:
                        best = (rho, float(w))
                w3 = best[1] if w3 is None else w3
            fusion3 = (1 - w3) * fixed2 + w3 * ccr_n

            hi = err > np.percentile(err, 90)
            k = int(0.2 * err.size)
            for name, trust in (("fixed2", 1 - fixed2), ("ccr_only", 1 - ccr_n),
                                ("fusion3_eq", 1 - (0.5 * fixed2 + 0.5 * ccr_n)),
                                ("fusion3_val", 1 - fusion3),
                                ("oracle", 1 - norm01(err))):
                srow = {"split": split, "idx": i, "subject": r["subject"], "z": r["z"],
                        "scheme": name,
                        "spearman": round(float(spearmanr((1 - trust).ravel(),
                                                         err.ravel()).statistic), 4),
                        "auroc": round(auroc_score(1 - trust, hi), 4)}
                drop = np.zeros(err.size, dtype=bool)
                drop[np.argsort((1 - trust).ravel())[:k]] = True
                keep = (~drop).reshape(err.shape)
                srow["gain20"] = round(
                    peak_signal_noise_ratio(g[keep], p[keep], data_range=g.max()) -
                    peak_signal_noise_ratio(g, p, data_range=g.max()), 3) if keep.sum() > 100 else np.nan
                schemes[name].append(srow)
                per_all.append(srow)
            if (i + 1) % 20 == 0 or i == len(rows) - 1:
                print(f"  [{split} {i+1}/{len(rows)}] w3={w3}", flush=True)

    # 聚合（仅 test）
    print("\n===== 测试集结果（T2→T1 主任务）=====")
    for name in ("fixed2", "ccr_only", "fusion3_eq", "fusion3_val", "oracle"):
        rows = [x for x in per_all if x["scheme"] == name and x["split"] == "test"]
        print(f"{name:14s} | spearman {np.mean([r['spearman'] for r in rows]):+.4f} "
              f"| auroc {np.nanmean([r['auroc'] for r in rows]):.4f} "
              f"| Δ20% {np.nanmean([r['gain20'] for r in rows]):+.3f} dB")
    summary["w3_ccr_weight_val"] = w3
    summary["schemes_test"] = {
        name: {"spearman": round(float(np.mean([r["spearman"] for r in per_all
                                                if r["scheme"] == name and r["split"] == "test"])), 4),
               "auroc": round(float(np.nanmean([r["auroc"] for r in per_all
                                                if r["scheme"] == name and r["split"] == "test"])), 4),
               "gain20": round(float(np.nanmean([r["gain20"] for r in per_all
                                                 if r["scheme"] == name and r["split"] == "test"])), 3)}
        for name in ("fixed2", "ccr_only", "fusion3_eq", "fusion3_val", "oracle")}
    run_dir = os.path.join(ROOT, "results",
                           f"{datetime.now():%Y-%m-%d}_cycle_signal_{args.task.replace('->','2')}")
    os.makedirs(run_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "per_slice.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_all[0].keys()))
        w.writeheader(); w.writerows(per_all)
    print("->", run_dir)


def norm01(m):
    hi = np.percentile(m, 99)
    return np.clip(m / hi, 0, 1).astype(np.float32) if hi > 1e-12 else np.zeros_like(m)


if __name__ == "__main__":
    main()
