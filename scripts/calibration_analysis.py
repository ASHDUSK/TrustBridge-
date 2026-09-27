"""不确定性校准分析：信任图作为"误差预测器"的校准质量（论文新增图表素材）。

内容（测试集，双方向）：
  1. 可靠性曲线：像素按信任分位（10 桶）分箱 → 每桶平均绝对误差（mean-norm 口径）
  2. 单调性得分：桶误差序列的单调递减程度（0-1）
  3. 分离比：最低信任桶误差 / 最高信任桶误差（越大越有判别力）
  4. ECE 风格：把 (1-trust) 线性映射到 [0, max_err] 后的加权 |预测-实际|（保守解释）

用法: python scripts/calibration_analysis.py --n 80
输出: results/2026-09-27_calibration/{summary.json, reliability_<task>.png}
"""
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scipy.stats import spearmanr  # noqa: E402

from trustbridge.engine import BridgeEngine, TASKS  # noqa: E402
from trustbridge.metrics import mean_norm  # noqa: E402
from trustbridge.pipeline import translate_single  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    run_dir = os.path.join(ROOT, "results",
                           f"{datetime.now():%Y-%m-%d}_calibration")
    os.makedirs(run_dir, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

    summary = {"date": datetime.now().isoformat(timespec="seconds"),
               "n_per_direction": args.n, "directions": {}}

    for task, ckpt in (("T2->T1", "checkpoints/self_t2t1.ckpt"),
                       ("T1->T2", "checkpoints/self_t1t2.ckpt")):
        eng = BridgeEngine(task, ckpt_path=ckpt)
        _, src_mod, tgt_mod, _ = TASKS[task]
        man = list(csv_reader(os.path.join(ROOT, "data", "ixi_slices",
                                           "manifest_v3.csv")))
        rows = [r for r in man if r["split"] == "test"]
        step = max(1, len(rows) // args.n)
        rows = rows[::step][: args.n]

        T, E = [], []
        for i, r in enumerate(rows):
            src = np.load(r["t2_path"] if src_mod == "T2" else r["t1_path"]).astype(np.float32)
            gt = np.load(r["t1_path"] if tgt_mod == "T1" else r["t2_path"]).astype(np.float32)
            out = translate_single(eng, src, n_ensemble=4,
                                   step_skip=1, seed=args.seed)
            trust, pred = out["result"].trust_map, out["pred01"]
            g, p = mean_norm(gt), mean_norm(pred)
            T.append(trust.ravel()); E.append(np.abs(g - p).ravel())
            if (i + 1) % 20 == 0 or i == len(rows) - 1:
                print(f"  [{task} {i+1}/{len(rows)}]", flush=True)

        t = np.concatenate(T); e = np.concatenate(E)
        # 分 10 桶（按信任降序）：桶 0 = 最可信
        order = np.argsort(t)
        bins = np.array_split(order, 10)
        bin_err = [float(e[b].mean()) for b in bins]
        bin_trust = [float(t[b].mean()) for b in bins]
        # 单调性：相邻桶误差递减的一致比例（信任升 → 误差降为正确方向）
        diffs = np.diff(bin_err)
        mono = float((diffs < 0).mean())
        sep = bin_err[0] / max(bin_err[-1], 1e-9)  # 分离比：最低信任桶误差 / 最高信任桶误差
        # ECE 风格：把 (1-trust) 缩放到误差量程后的加权绝对差
        pred_err = (1 - t) / max(1 - t.min(), 1e-9) * e.max()
        ece = float(np.abs(pred_err - e).mean())

        summary["directions"][task] = {
            "bin_trust": [round(v, 4) for v in bin_trust],
            "bin_abs_err": [round(v, 5) for v in bin_err],
            "monotonicity": round(mono, 3),
            "separation_ratio": round(sep, 2),
            "ece_style": round(ece, 5),
            "spearman_trust_err": round(float(spearmanr(t, e).statistic), 4),
        }
        print(f"  {task}: 单调性 {mono:.2f} | 分离比 {sep:.1f}x | ECE风格 {ece:.4f}")

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(range(1, 11), bin_err, marker="o", color="#2563eb")
        ax.set_xlabel("信任分位桶（1=最可信 → 10=最不可信）")
        ax.set_ylabel("桶内平均绝对误差")
        ax.set_title(f"{task} 可靠性曲线（测试集 {len(rows)} 片）")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, f"reliability_{task.replace('->','2')}.png"), dpi=150)
        plt.close(fig)

    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("->", run_dir)


def csv_reader(path):
    import csv
    return list(csv.DictReader(open(path, encoding="utf-8")))


if __name__ == "__main__":
    main()
