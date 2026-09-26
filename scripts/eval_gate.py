"""输入模态门控验证：训练（IXI val）+ 四组测试（IXI test 匹配/错配、SynthRAD MR、FeTS）。

用法: python scripts/eval_gate.py
输出: results/2026-09-27_modality_gate/{summary.json, per_slice.csv}
"""
import csv
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trustbridge.gate import check, fit_gate, gate_features  # noqa: E402


def load_dir(pattern, limit=None):
    fs = sorted(glob.glob(os.path.join(ROOT, pattern)))
    if limit:
        fs = fs[:limit]
    return [np.load(f).astype(np.float32) for f in fs]


def main():
    run_dir = os.path.join(ROOT, "results",
                           f"{datetime.now():%Y-%m-%d}_modality_gate")
    os.makedirs(run_dir, exist_ok=True)

    print("1) 训练（IXI val：T1 545 / T2 545）...")
    t1_val = load_dir("data/selfrdb_format_v3/T1/val/*.npy")
    t2_val = load_dir("data/selfrdb_format_v3/T2/val/*.npy")
    fit_info = fit_gate(t1_val, t2_val)
    print("  ", fit_info)

    suites = {}

    # A. IXI test 匹配/错配
    t1_test = load_dir("data/selfrdb_format_v3/T1/test/*.npy", 300)
    t2_test = load_dir("data/selfrdb_format_v3/T2/test/*.npy", 300)
    suites["A_ixi_test_T1_as_T1src(匹配)"] = [check(s, "T1") for s in t1_test]
    suites["B_ixi_test_T1_as_T2src(错配)"] = [check(s, "T2") for s in t1_test]
    suites["C_ixi_test_T2_as_T2src(匹配)"] = [check(s, "T2") for s in t2_test]
    suites["D_ixi_test_T2_as_T1src(错配)"] = [check(s, "T1") for s in t2_test]

    # C. SynthRAD 盆腔 MR（跨数据集 + 跨解剖，脑门控对照）
    mA = list(csv.DictReader(open(os.path.join(ROOT, "data/synthrad_pelvis_slices",
                                               "manifest_centerA.csv"), encoding="utf-8")))
    mC = list(csv.DictReader(open(os.path.join(ROOT, "data/synthrad_pelvis_slices",
                                               "manifest_centerC.csv"), encoding="utf-8")))
    mrA = [np.load(os.path.join(ROOT, r["mr_path"])).astype(np.float32)
           for r in mA[:150]]
    mrC = [np.load(os.path.join(ROOT, r["mr_path"])).astype(np.float32)
           for r in mC[:150]]
    suites["E_synthradA_T1w_as_T1src(匹配·脑门控)"] = [check(s, "T1") for s in mrA]
    suites["F_synthradC_T2w_as_T2src(匹配·脑门控)"] = [check(s, "T2") for s in mrC]
    # B2. 盆腔门控：用 SynthRAD 自身 A(T1w)/C(T2w) 训练（各取前 60 片训练）
    halfA, halfC = mrA[:60], mrC[:60]
    fit_gate(halfA, halfC, out_path=os.path.join(ROOT, "data", "modality_gate_pelvis.npz"))
    pelvis_gate = os.path.join(ROOT, "data", "modality_gate_pelvis.npz")
    pg = (lambda s, exp: check(s, exp, gate_path=pelvis_gate))
    suites["H_synthradA_T1w_盆腔门控(匹配)"] = [pg(s, "T1") for s in mrA[60:150]]
    suites["I_synthradC_T2w_盆腔门控(匹配)"] = [pg(s, "T2") for s in mrC[60:150]]
    suites["J_synthradA_T1w_喂T2源(错配·盆腔门控)"] = [pg(s, "T2") for s in mrA[60:150]]



    # D. FeTS 肿瘤切片（T2/FLAIR 类）
    fets = [np.load(f).astype(np.float32) / 255.0
            for f in sorted(glob.glob(os.path.join(ROOT, "data/brats_fets_slices",
                                                   "FeTS21*_z0*.npy")))[:150]
            if "label" not in os.path.basename(f)]
    suites["G_fets_T2like_as_T2src(匹配)"] = [check(s, "T2") for s in fets]

    summary = {"date": datetime.now().isoformat(timespec="seconds"),
               "train": fit_info, "suites": {}}
    per_rows = []
    for name, results in suites.items():
        n = len(results)
        ok = sum(1 for r in results if r["ok"])
        flagged = n - ok
        summary["suites"][name] = {
            "n": n, "match_rate": round(ok / n, 4),
            "flagged": flagged,
            "mean_prob_t2": round(float(np.mean([r["prob_t2"] for r in results
                                                 if r["prob_t2"] is not None])), 4),
        }
        for i, r in enumerate(results):
            per_rows.append({"suite": name, "idx": i, "ok": r["ok"],
                             "prob_t2": r["prob_t2"]})
        print(f"  {name}: 匹配率 {ok}/{n} = {ok/n:.1%}")

    with open(os.path.join(run_dir, "per_slice.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["suite", "idx", "ok", "prob_t2"])
        w.writeheader(); w.writerows(per_rows)
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("->", run_dir)


if __name__ == "__main__":
    main()
