# -*- coding: utf-8 -*-
"""崩盘性质甄别：官方权重在我们数据上的低 PSNR 是结构崩坏还是纯强度偏移？
测 NCC / SSIM / 逐片强度偏移量，区分两种失败模式。"""
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trustbridge.engine import BridgeEngine  # noqa: E402
from trustbridge.pipeline import translate_single  # noqa: E402
from trustbridge.metrics import mean_norm  # noqa: E402


def ncc(a, b, mask):
    a, b = a[mask].astype(np.float64), b[mask].astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-9))


def main(n=20, seed=2026):
    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_v3.csv"), encoding="utf-8")))
    rows = [r for r in man if r["split"] == "test"]
    step = max(1, len(rows) // n)
    rows = rows[::step][:n]
    out = []
    for label, task_id, ckpt in [
            ("official_13db", "T1->T2", os.path.join(ROOT, "checkpoints", "ixi_t1_t2.ckpt")),
            ("self_18db", "T1->T2*", os.path.join(ROOT, "checkpoints", "self_t1t2.ckpt"))]:
        eng = BridgeEngine(task_id, ckpt_path=ckpt)
        for r in rows:
            t1 = np.load(r["t1_path"]).astype(np.float32)
            t2 = np.load(r["t2_path"]).astype(np.float32)
            src, gt = t1, t2
            mask = src > 0.02
            o = translate_single(eng, src, n_ensemble=4, step_skip=1, seed=seed)
            pred = o["pred01"]
            if pred.shape != src.shape:
                S = max(pred.shape[0], src.shape[0])
                H, W = src.shape
                pt, pl = (pred.shape[0] - H) // 2, (pred.shape[1] - W) // 2
                pred = np.ascontiguousarray(pred[pt:pt + H, pl:pl + W])
            # 逐片最优线性强度映射（仅诊断用）
            g, p = gt[mask], pred[mask]
            k = float((g * p).sum() / max((p * p).sum(), 1e-9))
            g_, p_ = mean_norm(gt), mean_norm(pred)
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity
            psnr_lin = float(peak_signal_noise_ratio(g_[mask], (k * p_)[mask],
                                                     data_range=g_[mask].max()))
            ssim = float(structural_similarity(mean_norm(gt), k * mean_norm(pred),
                                               data_range=1.0) * 100)
            out.append({"model": label, "subject": r["subject"], "z": r["z"],
                        "ncc": round(ncc(gt, pred, mask), 4),
                        "linear_gain_k": round(k, 3),
                        "psnr_after_linearcorr": round(psnr_lin, 2),
                        "ssim_pct": round(ssim, 2)})
        print(label, "done", flush=True)
    run_dir = os.path.join(ROOT, "results", f"{time.strftime('%Y-%m-%d')}_collapse_ncc")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "per_case.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    summary = {}
    for label in ("official_13db", "self_18db"):
        rs = [q for q in out if q["model"] == label]
        summary[label] = {
            "n": len(rs),
            "ncc": round(float(np.mean([q["ncc"] for q in rs])), 4),
            "psnr_after_linearcorr": round(float(np.mean([q["psnr_after_linearcorr"] for q in rs])), 2),
            "ssim_pct": round(float(np.mean([q["ssim_pct"] for q in rs])), 2),
            "mean_linear_gain": round(float(np.mean([q["linear_gain_k"] for q in rs])), 3),
        }
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("->", run_dir)


if __name__ == "__main__":
    main()
