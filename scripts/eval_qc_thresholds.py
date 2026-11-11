"""QC 阈值论证：Trust Score 三级门槛（75/55）的操作点分析。

问题：应用的三级门槛 PASS>=75 / REVIEW 55-75 / REJECT<55 此前无数据支撑。
方法：汇集 UQ 基线实验（IXI 双方向）与 BraTS 体区复检的逐片数据，
      以切片体区 PSNR 的任务内下四分位为"问题切片"，扫描阈值 t：
        - 召回率   P(问题切片 | score < t)
        - 精确率   P(score < t | 问题切片)
      输出 75/55 处的实际含义，并给出数据驱动建议值。

用法: python scripts/eval_qc_thresholds.py
输入: results/*_uq_baselines_T1_T2/per_slice.csv（及 T2_T1）、
      results/*_tumor_sparsify_body/per_slice.csv
输出: results/<date>_qc_thresholds/{thresholds.csv, summary.json}
"""
import csv
import glob
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(task_dirs):
    rows = []
    for d in task_dirs:
        for p in glob.glob(os.path.join(ROOT, "results", d, "per_slice.csv")):
            for r in csv.DictReader(open(p, encoding="utf-8")):
                if r.get("body_psnr") and r.get("trust_score"):
                    rows.append({"task": r.get("task", d), "psnr": float(r["body_psnr"]),
                                 "score": float(r["trust_score"])})
    return rows


def main():
    rows = load(["2026-11-12_slice_scores_T1_T2", "2026-11-12_slice_scores_T2_T1",
                 "2026-11-12_tumor_sparsify_body"])
    if not rows:
        print("找不到逐片数据（先跑 eval_uq_baselines.py / eval_tumor_sparsify.py）")
        return
    # 任务内标准化：PSNR 低于该任务下四分位 = 问题切片
    tasks = sorted({r["task"] for r in rows})
    for t in tasks:
        ps = np.array([r["psnr"] for r in rows if r["task"] == t])
        q25 = np.percentile(ps, 25)
        for r in rows:
            if r["task"] == t:
                r["problem"] = r["psnr"] < q25
    scores = np.array([r["score"] for r in rows])
    problems = np.array([r["problem"] for r in rows])
    print(f"共 {len(rows)} 片 / {len(tasks)} 组；问题切片 {problems.mean():.1%}")

    out = []
    for t in range(40, 96):
        flagged = scores < t
        if flagged.sum() == 0:
            continue
        recall = float(flagged[problems].mean())          # 问题切片被拦截比例
        precision = float(problems[flagged].mean()) if flagged.any() else 0.0
        out.append({"thresh": t, "flagged_pct": round(float(flagged.mean()), 4),
                    "recall": round(recall, 4), "precision": round(precision, 4)})

    def at(t):
        r = next((x for x in out if x["thresh"] == t), None)
        return r or {"thresh": t}

    # 数据驱动建议：精确率最高的 REJECT 阈值（拦截率 <= 20%）
    cand = [x for x in out if x["flagged_pct"] <= 0.20]
    best = max(cand, key=lambda x: (x["precision"], x["recall"])) if cand else None

    summary = {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
               "n_slices": len(rows), "tasks": tasks,
               "problem_def": "任务内体区PSNR下四分位",
               "at_75_PASS": at(75), "at_55_REVIEW": at(55),
               "suggested_reject_thresh": best,
               "score_p50": round(float(np.percentile(scores, 50)), 1),
               "score_p10": round(float(np.percentile(scores, 10)), 1)}
    run_dir = os.path.join(ROOT, "results", f"{time.strftime('%Y-%m-%d')}_qc_thresholds")
    os.makedirs(run_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "thresholds.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("->", run_dir)


if __name__ == "__main__":
    main()
