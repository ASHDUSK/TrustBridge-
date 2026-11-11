# -*- coding: utf-8 -*-
"""体区口径十分箱可靠性数据（替换整图 35-73x 版）：IXI 双方向 60 片。
输出: results/<date>_body_decile/per_case.csv（每片各十分箱平均绝对误差）+
      summary.json（跨片中位曲线，论文 fig 用的就是这组数）。
"""
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trustbridge.engine import BridgeEngine, TASKS  # noqa: E402
from trustbridge.pipeline import translate_single, get_sampler  # noqa: E402
from trustbridge.sampler import _pct_norm  # noqa: E402



def align_map(m, ref):
    """信任图对齐到源尺寸：小于模型尺寸则中心裁剪，大于则双线性放大（同 pipeline）。"""
    S = m.shape[0]; H, W = ref.shape
    if H <= S and W <= S:
        pt, pl = (S - H) // 2, (S - W) // 2
        return np.ascontiguousarray(m[pt:pt + H, pl:pl + W])
    from skimage.transform import resize
    return resize(m, (H, W), order=1, preserve_range=True).astype(np.float32)

def main(n=60, seed=2026):
    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_v3.csv"), encoding="utf-8")))
    rows_all = [r for r in man if r["split"] == "test"]
    step = max(1, len(rows_all) // n)
    rows_all = rows_all[::step][:n]
    run_dir = os.path.join(ROOT, "results", f"{time.strftime('%Y-%m-%d')}_body_decile")
    os.makedirs(run_dir, exist_ok=True)

    out = []
    for task_id in ("T1->T2*", "T2->T1*"):
        eng = BridgeEngine(task_id)
        _, src_mod, _, _ = TASKS[task_id]
        for i, r in enumerate(rows_all):
            t1 = np.load(r["t1_path"]).astype(np.float32)
            t2 = np.load(r["t2_path"]).astype(np.float32)
            src, gt = (t2, t1) if src_mod == "T2" else (t1, t2)
            mask = src > 0.02
            o = translate_single(eng, src, n_ensemble=4, step_skip=1, seed=seed)
            res, pred = o["result"], o["pred01"]
            # 论文口径融合（CReF 0.6 / 集成 0.4），从原始图重算，不依赖 sampler 默认
            smp = get_sampler(eng)
            R_n = _pct_norm(smp.rcrf_raw_map)
            U_n = _pct_norm(smp.unc_raw_map)
            tr = np.clip(1 - 0.6 * R_n - 0.4 * U_n, 0, 1)
            tr = align_map(tr, src)
            err = np.abs(gt - pred)
            tr_m, er_m = tr[mask], err[mask]
            order = np.argsort(tr_m)              # 低信任在前
            bins = np.array_split(order, 10)
            row = {"task": task_id, "subject": r["subject"], "z": r["z"]}
            for b_i, idx in enumerate(bins):
                row[f"bin{b_i+1}"] = round(float(er_m[idx].mean()), 5)
            out.append(row)
            if (i + 1) % 10 == 0:
                print(f"  [{task_id} {i+1}/{len(rows_all)}]", flush=True)

    with open(os.path.join(run_dir, "per_case.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    summary = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "protocol": "within-body deciles, 10 bins, N=4",
               "curves": {}}
    for task_id in ("T1->T2*", "T2->T1*"):
        rs = [q for q in out if q["task"] == task_id]
        med = [float(np.median([q[f"bin{b}"] for q in rs])) for b in range(1, 11)]
        summary["curves"][task_id] = {"median_abs_err_per_bin": [round(v, 4) for v in med],
                                      "separation_low_over_high": round(med[0] / max(med[-1], 1e-9), 2)}
    import json
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(summary["curves"])
    print("->", run_dir)


if __name__ == "__main__":
    main()
