# -*- coding: utf-8 -*-
"""Gate 3 · 患者级配对 bootstrap 统计（硬性标准 C）。

数据源（全部已有 CSV，纯离线）：
  1) results/<date>_uq_baselines_{T1_T2,T2_T1}_self/per_slice.csv
     —— 自训模型 8 方法 × 60 片（8 被试）：配对 ΔAUROC / Δdpsnr20 CI
  2) results/<date>_uq_baselines_T2_T1_official/per_slice.csv
     —— 官方权重 8 方法：官方 T2→T1 各信号 CI 对照 0.5
  3) results/<date>_tumor_sparsify_body/per_slice.csv —— 肿瘤 low-trust AUROC 患者 CI
  4) results/<date>_crossinst_brats/per_case.csv —— 跨机构两组患者 CI
  5) results/maps_full/synthrad_*/derived.json —— SynthRAD（患者数受限，如实报告）

输出: results/<date>_patient_stats/summary.json + forest 数据 CSV
"""
import csv
import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B = 5000


def load_group(path, config):
    """返回 {subject: [auroc...]}（该 config 的逐片 AUROC 按患者聚合）。"""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    d = {}
    for r in rows:
        if r["config"] == config:
            d.setdefault(r["subject"], []).append(float(r["auroc"]))
    return d


def paired_bootstrap(sub_a, sub_b, B=B, seed=2026):
    """subject→[slice aurocs] 两方法配对 bootstrap。
    返回: 双口径 {macro: (点估计, CI), pooled_cluster: (点估计, CI)} 的 Δ 与各自 CI。"""
    rng = np.random.RandomState(seed)
    subs = sorted(set(sub_a) & set(sub_b))
    S = len(subs)
    macro_a = np.array([np.mean(sub_a[s]) for s in subs])
    macro_b = np.array([np.mean(sub_b[s]) for s in subs])
    slices_a = np.concatenate([sub_a[s] for s in subs])
    slices_b = np.concatenate([sub_b[s] for s in subs])
    # 逐片归属（用于 cluster 重采样）
    owner = np.concatenate([[i] * len(sub_a[s]) for i, s in enumerate(subs)])

    d_macro = macro_a - macro_b
    d_pooled = slices_a.mean() - slices_b.mean()

    bm, bp = [], []
    for _ in range(B):
        idx = rng.randint(0, S, S)                 # 患者级有放回
        bm.append(macro_a[idx].mean() - macro_b[idx].mean())
        sel = np.concatenate([np.where(owner == i)[0] for i in idx])
        bp.append(slices_a[sel].mean() - slices_b[sel].mean())
    ci = lambda arr: (round(float(np.percentile(arr, 2.5)), 4),
                      round(float(np.percentile(arr, 97.5)), 4))
    return {"n_subjects": S, "subjects": subs,
            "subject_macro": {"delta": round(float(d_macro.mean()), 4),
                              "per_subject": [round(x, 4) for x in d_macro],
                              "ci95": ci(np.array(bm))},
            "cluster_pooled": {"delta": round(float(d_pooled), 4),
                               "ci95": ci(np.array(bp))}}


def one_sided_vs_chance(sub_values, chance=0.5, B=B, seed=2026):
    rng = np.random.RandomState(seed)
    vals = np.array(sub_values)
    obs = float(vals.mean())
    bs = np.array([vals[rng.randint(0, len(vals), len(vals))].mean() for _ in range(B)])
    ci = (round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4))
    covers = ci[0] <= chance <= ci[1]
    return {"n": len(vals), "mean": round(obs, 4), "ci95": ci,
            "ci_covers_chance": bool(covers)}


def main():
    out = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "B": B, "seed": 2026}
    R = os.path.join(ROOT, "results")

    # ---- 1) 自训模型：配对差（融合 vs 仅集成 / CReF-N1 / TTA / 噪声）----
    for task, tag in (("T1->T2", "T1_T2"), ("T2->T1", "T2_T1")):
        p = os.path.join(R, f"2026-11-12_uq_baselines_{tag}_self", "per_slice.csv")
        base = load_group(p, "ours_fused")
        for cfg in ("ours_unc_n4", "ours_rcrf_n1", "tta_disp", "noise_disp"):
            other = load_group(p, cfg)
            out[f"self_{tag}:fused_vs_{cfg}"] = paired_bootstrap(base, other)

    # ---- 2) 官方权重：T2→T1 各信号 vs 0.5（"全盲"主张的 CI 化）----
    p = os.path.join(R, "2026-11-12_uq_baselines_T2_T1_official", "per_slice.csv")
    for cfg in ("ours_fused", "ours_unc_n4", "ours_rcrf_n1", "tta_disp", "noise_disp"):
        d = load_group(p, cfg)
        vals = [np.mean(v) for v in d.values()]
        out[f"official_T2T1:{cfg}_vs_chance"] = one_sided_vs_chance(vals)

    # ---- 3) 肿瘤 low-trust AUROC 患者 CI ----
    p = os.path.join(R, "2026-11-12_tumor_sparsify_body", "per_slice.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    bypat = {}
    for r in rows:
        bypat.setdefault(r["patient"], []).append(float(r["auroc"]))
    out["tumor_lowtrust_vs_0.5"] = one_sided_vs_chance([np.mean(v) for v in bypat.values()])

    # ---- 4) 跨机构两组患者 CI ----
    p = os.path.join(R, "2026-11-12_crossinst_brats", "per_case.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for set_name in ("B_same_collection_heldout", "C_cross_institution_cbica"):
        bypat = {}
        for r in rows:
            if r["set"] == set_name:
                bypat.setdefault(r["patient"], []).append(float(r["auroc"]))
        vals = [np.mean(v) for v in bypat.values()]
        out[f"crossinst:{set_name}"] = one_sided_vs_chance(vals)

    run_dir = os.path.join(R, time.strftime("%Y-%m-%d") + "_patient_stats")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # 森林图数据（患者级 fused AUROC 分布）
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("->", run_dir)


if __name__ == "__main__":
    main()
