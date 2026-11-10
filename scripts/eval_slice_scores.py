"""切片级 Trust Score + 体区 PSNR 采集（供 QC 阈值分析，与消融/UQ 同切片选择）。

只跑融合推理（N=4），每片存 trust_score（体区均值×100）与 body_psnr。
用法: python scripts/eval_slice_scores.py --task "T1->T2" [--n 60]
输出: results/<date>_slice_scores_<task>/per_slice.csv
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from skimage.metrics import peak_signal_noise_ratio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trustbridge.engine import BridgeEngine, TASKS  # noqa: E402
from trustbridge.pipeline import translate_single  # noqa: E402
from trustbridge.metrics import mean_norm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T1->T2")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_v3.csv"), encoding="utf-8")))
    rows = [r for r in man if r["split"] == "test"]
    step = max(1, len(rows) // args.n)
    rows = rows[::step][: args.n]

    eng = BridgeEngine(args.task, ckpt_path=args.ckpt)  # 显式自训 ckpt
    _, src_mod, _, _ = TASKS[args.task]
    run_dir = os.path.join(ROOT, "results",
                           f"{time.strftime('%Y-%m-%d')}_slice_scores_{args.task.replace('->', '_')}")
    os.makedirs(run_dir, exist_ok=True)

    per = []
    for i, r in enumerate(rows):
        t1 = np.load(r["t1_path"]).astype(np.float32)
        t2 = np.load(r["t2_path"]).astype(np.float32)
        src, gt = (t2, t1) if src_mod == "T2" else (t1, t2)
        mask = src > 0.02
        out = translate_single(eng, src, n_ensemble=4, step_skip=1, seed=args.seed)
        res, pred = out["result"], out["pred01"]
        tr = res.trust_map
        if tr.shape != src.shape:
            S = tr.shape[0]; H, W = src.shape
            pt, pl = (S - H) // 2, (S - W) // 2
            tr = np.ascontiguousarray(tr[pt:pt + H, pl:pl + W])
            pred = np.ascontiguousarray(pred[pt:pt + H, pl:pl + W])
        g, p = mean_norm(gt), mean_norm(pred)
        psnr = float(peak_signal_noise_ratio(g[mask], p[mask], data_range=g[mask].max()))
        per.append({"subject": r["subject"], "z": r["z"],
                    "trust_score": round(float(tr[mask].mean() * 100), 1),
                    "body_psnr": round(psnr, 2)})
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            print(f"  [{i+1}/{len(rows)}] {(time.time())}", flush=True)

    with open(os.path.join(run_dir, "per_slice.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "z", "trust_score", "body_psnr"])
        w.writeheader(); w.writerows(per)
    import json
    sc = np.array([q["trust_score"] for q in per]); ps = np.array([q["body_psnr"] for q in per])
    json.dump({"n": len(per), "score_mean": round(float(sc.mean()), 1),
               "score_std": round(float(sc.std()), 1),
               "psnr_mean": round(float(ps.mean()), 2),
               "corr_pearson": round(float(np.corrcoef(sc, ps)[0, 1]), 4),
               "corr_spearman_src": None},
              open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("->", run_dir)


if __name__ == "__main__":
    main()
