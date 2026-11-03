# -*- coding: utf-8 -*-
"""Evaluate Your Own Model —— 体掩码不确定性评估一键脚本（社区版）。

把你自己的翻译模型（任意框架）接进来，在配对测试集上算出论文同款
体掩码指标：AUROC（低信任→高误差十分位）、ΔPSNR@20%（稀疏化增益）、
十分位分离、体区 PSNR。另附整图口径对照，直观展示"口径陷阱"。

你的模型只需提供一个 predict(src01) -> pred01 的函数（[0,1] float32 HxW）
和一张与 pred 同形的 trust/suspicion 图（低=可信 或 高=可疑，二选一声明）。

用法:
  python scripts/evaluate_your_model.py --demo          # 用本仓库自训模型演示
  python scripts/evaluate_your_model.py --my-module my_pkg.my_model
      （模块需提供 predict(src01)->pred01 与 trust_map_or_suspicion(src01,pred01)->2D ndarray，
        以及可选的 SUSPICION=True/False 声明）
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


def sparsify(gt, pred, trust, mask, drop_pct):
    g, p = mean_norm(gt), mean_norm(pred)
    full = peak_signal_noise_ratio(g[mask], p[mask], data_range=g[mask].max())
    tr = trust[mask]
    keep = tr >= np.percentile(tr, drop_pct)
    if keep.sum() < 100:
        return 0.0
    return float(peak_signal_noise_ratio(g[mask][keep], p[mask][keep],
                                         data_range=g[mask].max()) - full)


def decile_sep(trust, err, mask):
    tr, er = trust[mask], err[mask]
    lo = er[tr <= np.percentile(tr, 10)].mean()
    hi = er[tr >= np.percentile(tr, 90)].mean()
    return float(lo / max(hi, 1e-9))


def whole_image_auroc(trust, err):
    """整图口径对照：展示背景空气如何虚高 AUROC（论文的口径陷阱）。"""
    return auroc_low_vs_higherr(trust, err)   # 无 mask，全图


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="用本仓库自训 T1->T2 模型演示")
    ap.add_argument("--my-module", default=None,
                    help="你的模型模块（import 路径），见文件头注释的接口约定")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    if args.demo:
        from trustbridge.engine import BridgeEngine
        eng = BridgeEngine("T1->T2*", ckpt_path=os.path.join(ROOT, "checkpoints", "self_t1t2.ckpt"))

        def predict(src01):
            return eng, translate(eng, src01)

        def translate(eng, src01):
            from trustbridge.pipeline import translate_single
            o = translate_single(eng, src01, n_ensemble=4, step_skip=1, seed=args.seed)
            return o["pred01"], o["result"].trust_map
        src_fn = lambda r: r["t1_path"]
        gt_fn = lambda r: r["t2_path"]
        predict_pair = lambda src01: translate(eng, src01)
    elif args.my_module:
        mod = __import__(args.my_module, fromlist=["*"])
        predict_pair = mod.predict_pair      # (src01) -> (pred01, trust_like_2d)
        suspicion = bool(getattr(mod, "SUSPICION", False))
        src_fn = mod.src_path
        gt_fn = mod.gt_path
    else:
        ap.error("请指定 --demo 或 --my-module")

    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_v3.csv"), encoding="utf-8")))
    rows = [r for r in man if r["split"] == "test"]
    step = max(1, len(rows) // args.n)
    rows = rows[::step][:args.n]

    recs = []
    for i, r in enumerate(rows):
        src = np.load(src_fn(r)).astype(np.float32)
        gt = np.load(gt_fn(r)).astype(np.float32)
        mask = src > 0.02
        pred, sig = predict_pair(src)
        if pred.shape != src.shape:
            H, W = src.shape
            pt = (pred.shape[0] - H) // 2
            pl = (pred.shape[1] - W) // 2
            pred = np.ascontiguousarray(pred[pt:pt + H, pl:pl + W])
            sig = np.ascontiguousarray(sig[pt:pt + H, pl:pl + W])
        if args.demo or not suspicion:
            trust = sig                      # 低=可信
        else:
            trust = 1.0 - sig                # 高=可疑 → 转 trust
        err = np.abs(gt - pred)
        g, p = mean_norm(gt), mean_norm(pred)
        psnr = float(peak_signal_noise_ratio(g[mask], p[mask], data_range=g[mask].max()))
        recs.append({
            "subject": r["subject"], "z": r["z"], "body_psnr": round(psnr, 2),
            "auroc_body": round(auroc_low_vs_higherr(trust, err), 4),
            "auroc_whole_image": round(whole_image_auroc(trust, err), 4),
            "dpsnr20_body": round(sparsify(gt, pred, trust, mask, 20), 3),
            "decile_sep_body": round(decile_sep(trust, err, mask), 2),
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(rows)}]", flush=True)

    run_dir = os.path.join(ROOT, "results", f"{time.strftime('%Y-%m-%d')}_evaluate_your_model")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "per_case.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(recs)
    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_slices": len(recs),
        "within_body": {k: {"mean": round(float(np.nanmean([q[k] for q in recs])), 4),
                            "std": round(float(np.nanstd([q[k] for q in recs])), 4)}
                        for k in ("body_psnr", "auroc_body", "dpsnr20_body", "decile_sep_body")},
        "whole_image_trap": {"auroc_whole_image": round(
            float(np.nanmean([q["auroc_whole_image"] for q in recs])), 4),
            "note": "整图口径 vs 体区口径的差 = 口径陷阱效应（论文 §4.1）"},
    }
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("->", run_dir)


if __name__ == "__main__":
    main()
