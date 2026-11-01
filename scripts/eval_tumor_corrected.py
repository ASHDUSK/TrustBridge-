"""BraTS2018 病灶-信任分析（修正版）：AUROC 限定体掩码内（排除空气负例）。

修正点：旧评估在整图上计算 AUROC，背景空气（信任≈1，非瘤负例）把
"低信任→肿瘤" 的 AUROC 虚高为 0.871。本脚本限定在体掩码（FLAIR>0.02）内
计算，并同场输出两个方向：低信任→肿瘤 与 高信任→肿瘤。

用法: python scripts/eval_tumor_corrected.py [--n 60]
输出: results/2026-10-31_tumor_trust_corrected/
"""
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
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "brats2018_slices",
                                                "manifest.csv"), encoding="utf-8")))
    man_t = [r for r in man if float(r["tumor_frac"]) > 0.005]
    step = max(1, len(man_t) // args.n)
    sel = man_t[::step][: args.n]
    print(f"评估 {len(sel)} 个含瘤切片（tumor_frac>0.005）")

    eng = BridgeEngine("BRATS_FLAIR->T2",
                       ckpt_path=os.path.join(ROOT, "checkpoints", "brats_flair_t2.ckpt"))
    run_dir = os.path.join(ROOT, "results", "2026-10-31_tumor_trust_corrected")
    os.makedirs(run_dir, exist_ok=True)

    per = []
    t0 = time.time()
    for i, r in enumerate(sel):
        flair = np.load(r["flair_path"]).astype(np.float32)
        t2gt = np.load(r["t2_path"]).astype(np.float32)
        seg = np.load(r["seg_path"]).astype(bool)
        body = flair > 0.02

        out = translate_single(eng, flair, n_ensemble=4, step_skip=1, seed=args.seed)
        res = out["result"]; pred = out["pred01"]
        trust = res.trust_map
        if trust.shape != flair.shape:
            S = trust.shape[0]; H, W = flair.shape
            pt, pl = (S - H) // 2, (S - W) // 2
            crop = lambda m: np.ascontiguousarray(m[pt:pt + H, pl:pl + W])
            trust, pred = crop(trust), crop(pred)
        err = np.abs(t2gt - pred)

        t_in = float(trust[seg].mean())
        t_out = float(trust[body & ~seg].mean())
        e_in = float(err[seg].mean())
        e_out = float(err[body & ~seg].mean())
        au_low = auroc((1.0 - trust)[body], seg[body])   # 低信任→肿瘤（体掩码内，正确口径）
        au_high = auroc(trust[body], seg[body])          # 高信任→肿瘤（方向对照）

        per.append({"patient": r["patient"], "z": r["z"], "tumor_frac": float(r["tumor_frac"]),
                    "t_in": round(t_in, 4), "t_out": round(t_out, 4),
                    "e_in": round(e_in, 5), "e_out": round(e_out, 5),
                    "auroc_lowtrust": round(au_low, 4), "auroc_hightrust": round(au_high, 4)})
        if (i + 1) % 10 == 0 or i == len(sel) - 1:
            print(f"  [{i+1}/{len(sel)}] AUROC(低信任→肿瘤) {np.mean([p['auroc_lowtrust'] for p in per]):.3f}",
                  flush=True)

    import json
    au_low = np.array([p["auroc_lowtrust"] for p in per])
    au_high = np.array([p["auroc_hightrust"] for p in per])
    e_in = np.array([p["e_in"] for p in per]); e_out = np.array([p["e_out"] for p in per])
    t_in = np.array([p["t_in"] for p in per]); t_out = np.array([p["t_out"] for p in per])

    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_slices": len(per), "seed": args.seed,
        "note": "修正版：AUROC 限定体掩码内（排除空气负例）。"
                "旧口径（整图含空气）0.871 系方向标反+空气负例虚高。",
        "auroc_lowtrust_to_tumor": {"mean": round(float(au_low.mean()), 4),
                                    "std": round(float(au_low.std()), 4)},
        "auroc_hightrust_to_tumor": {"mean": round(float(au_high.mean()), 4)},
        "trust_in": round(float(t_in.mean()), 4), "trust_out": round(float(t_out.mean()), 4),
        "err_in": round(float(e_in.mean()), 5), "err_out": round(float(e_out.mean()), 5),
        "err_ratio": round(float(e_in.mean() / max(e_out.mean(), 1e-9)), 3),
        "slices_with_higher_err_inside": round(float((e_in > e_out).mean()), 3),
    }
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "per_slice.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader()
        for p in per:
            w.writerow(p)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("->", run_dir)


if __name__ == "__main__":
    main()
