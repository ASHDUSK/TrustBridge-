"""TTA（测试时增强）不确定性基线：与本文信任图同预算对照。

协议：对每张测试切片，运行 4 次平移不变的输入增强（原始/小角度旋转±3°/亮度×0.95），
将各次预测旋转回原图框架后取逐像素标准差作为 TTA 不确定性；体掩码内计算
"TTA 低不确定性→高误差像素" 的 AUROC 与逐片 Spearman，与本文信任图（同切片、同 N=4）
直接对照。数据：IXI T1→T2 测试集 30 片（与主实验同分布）。

用法: python scripts/eval_tta_baseline.py [--n 30]
输出: results/2026-10-31_tta_baseline/{per_slice.csv,summary.json}
"""
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scipy.ndimage import rotate as ndrotate  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from trustbridge.engine import BridgeEngine  # noqa: E402
from trustbridge.pipeline import translate_single  # noqa: E402


def auroc(score, pos):
    s = score.ravel().astype(np.float64)
    p = pos.ravel().astype(bool)
    npos = int(p.sum()); nneg = s.size - npos
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1)
    return float((ranks[p].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T1->T2")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    from trustbridge.engine import BridgeEngine as BE
    eng = BE(args.task, ckpt_path=os.path.join(
        ROOT, "checkpoints", "self_t1t2.ckpt" if args.task == "T1->T2" else "self_t2t1.ckpt"))

    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_v3.csv"), encoding="utf-8")))
    rows = [r for r in man if r["split"] == "test"]
    step = max(1, len(rows) // args.n)
    rows = rows[::step][: args.n]

    per = []
    t0 = time.time()
    for i, r in enumerate(rows):
        src = np.load(r["t2_path"] if args.task == "T1->T2" else r["t1_path"]).astype(np.float32)
        gt = np.load(r["t1_path"] if args.task == "T1->T2" else r["t2_path"]).astype(np.float32)
        body = src > 0.02

        # 4 次增强推理：原始 / rot+3 / rot-3 / 亮度×1.05（预测均映射回原框架）
        preds = []
        for j, aug in enumerate(["orig", "rot+", "rot-", "bright"]):
            inp = src
            if aug == "rot+":
                inp = ndrotate(src, 3, reshape=False, order=1)
            elif aug == "rot-":
                inp = ndrotate(src, -3, reshape=False, order=1)
            elif aug == "bright":
                inp = np.clip(src * 1.05, 0, 1)
            out = translate_single(eng, np.ascontiguousarray(inp), n_ensemble=1,
                                   step_skip=1, seed=args.seed + j)
            pred = out["pred01"]
            if aug in ("rot+", "rot-"):
                pred = ndrotate(pred, -3 if aug == "rot+" else 3, reshape=False, order=1)
            preds.append(pred)
        preds = np.stack(preds)
        tta_std = preds.std(axis=0)                    # TTA 不确定性
        err = np.abs(gt - preds.mean(axis=0))          # 均值预测的误差

        au = auroc((1.0 - tta_std / (np.percentile(tta_std, 99) + 1e-9))[body], (err > np.percentile(err[body], 90))[body])
        rho = float(spearmanr(tta_std[body], err[body]).statistic)
        per.append({"subject": r["subject"], "z": r["z"],
                    "auroc_tta": round(au, 4), "spearman_tta": round(rho, 4)})
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            print(f"  [{i+1}/{len(rows)}] TTA AUROC {np.mean([p['auroc_tta'] for p in per]):.3f}",
                  flush=True)

    aus = np.array([p["auroc_tta"] for p in per])
    rhos = np.array([p["spearman_tta"] for p in per])
    summary = {
        "task": args.task, "n_slices": len(per), "seed": args.seed,
        "protocol": "TTA: orig/rot+3/rot-3/bright x1.05, N=4, within-body AUROC",
        "auroc_tta_mean": round(float(aus.mean()), 4),
        "auroc_tta_std": round(float(aus.std()), 4),
        "spearman_tta_mean": round(float(rhos.mean()), 4),
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    run = os.path.join(ROOT, "results", "2026-10-31_tta_baseline")
    os.makedirs(run, exist_ok=True)
    json.dump(summary, open(os.path.join(run, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(os.path.join(run, "per_slice.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader()
        for p in per:
            w.writerow(p)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("->", run)


if __name__ == "__main__":
    main()
