"""自适应采样对比实验：同预算下 uniform vs 信任引导 adaptive（改进路线图阶段一）。

配置（同 seed，公平对比）：
  A. uniform N=4        基线（4 链均匀）
  B. adaptive 2+2       n_base=2 + 低信任块追加 m_extra=2（总预算 ≤4，与 A 对齐）
  C. uniform N=2        一半预算参考（同 base 成本）

验收（docs/07 改进路线图）：同预算 PSNR +0.3dB 或 同质量耗时 −30%。
输出: results/<date>_ixi_adaptive/{config.json,per_slice.csv,summary.json,figures/*.png}
用法: python scripts/eval_adaptive.py [--n 60] [--smoke]
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # noqa: E402
from trustbridge.engine import BridgeEngine                                 # noqa: E402
from trustbridge.sampler import TrustSampler, AdaptiveTrustSampler          # noqa: E402
from trustbridge.metrics import mean_norm                                   # noqa: E402


def psnr_of(g, p):
    return float(peak_signal_noise_ratio(g, p, data_range=g.max()))


def ssim_of(g, p):
    return float(structural_similarity(g, p, data_range=g.max()) * 100)


def run_config(eng, sampler, img01, gt01, n_ens, seed, adaptive=False, **kw):
    """统一跑一个配置：返回 (pred01, trust_map, elapsed, extra)。"""
    y = eng.preprocess(img01)
    t0 = time.time()
    if adaptive:
        res = sampler.sample_adaptive(y, seed=seed, **kw)
    else:
        res = sampler.sample(y, n_ensemble=n_ens, step_skip=1, seed=seed)
    elapsed = time.time() - t0
    pred = eng.postprocess(res.pred, img01.shape)
    g, p = mean_norm(gt01), mean_norm(pred)
    extra = {}
    if adaptive:
        extra["used_extra"] = bool(res.members.shape[0] > kw.get("n_base", 2))
        extra["n_members"] = int(res.members.shape[0])
        extra["recursions_mean"] = float(np.mean([sum(c) / len(c) for c in res.recursions_used]))
    return dict(psnr=psnr_of(g, p), ssim=ssim_of(g, p), trust_map=res.trust_map,
                elapsed=elapsed, pred=pred, **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T2->T1*")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--block-thr", type=float, default=0.01)
    ap.add_argument("--r-max", type=int, default=4)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--n-base", type=int, default=2)
    ap.add_argument("--m-extra", type=int, default=2)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    import csv
    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices", "manifest_v3.csv"), encoding="utf-8")))
    rows = [r for r in man if r["split"] == "test"]
    step = max(1, len(rows) // args.n)
    rows = rows[::step][: args.n]
    if args.smoke:
        rows = rows[:5]
    print(f"自适应采样对比：{len(rows)} 片 | {args.task}")

    run_dir = os.path.join(ROOT, "results",
                           f"{datetime.now():%Y-%m-%d}_ixi_adaptive"
                           + (f"_{args.tag}" if args.tag else "")
                           + ("_smoke" if args.smoke else ""))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)

    eng = BridgeEngine(args.task, ckpt_path=args.ckpt)
    uni4 = TrustSampler(eng)
    ada = AdaptiveTrustSampler(eng)
    uni2 = TrustSampler(eng)

    per = []
    t_start = time.time()
    for i, r in enumerate(rows):
        src = np.load(r["t2_path"]).astype(np.float32)
        gt = np.load(r["t1_path"]).astype(np.float32)

        A = run_config(eng, uni4, src, gt, n_ens=4, seed=args.seed)
        B = run_config(eng, ada, src, gt, n_ens=4, seed=args.seed,
                       adaptive=True, n_base=args.n_base, m_extra=args.m_extra,
                       r_max=args.r_max, block=args.block, block_thr=args.block_thr)
        C = run_config(eng, uni2, src, gt, n_ens=2, seed=args.seed)

        per.append({
            "idx": i,
            "psnr_uniform4": round(A["psnr"], 3), "psnr_adaptive22": round(B["psnr"], 3),
            "psnr_uniform2": round(C["psnr"], 3),
            "ssim_uniform4": round(A["ssim"], 2), "ssim_adaptive22": round(B["ssim"], 2),
            "time_uniform4": round(A["elapsed"], 2), "time_adaptive22": round(B["elapsed"], 2),
            "time_uniform2": round(C["elapsed"], 2),
            "adaptive_used_extra": B.get("used_extra"), "adaptive_n_members": B.get("n_members"),
            "adaptive_recursions_mean": round(B.get("recursions_mean", 0), 2),
        })
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            print(f"  [{i+1}/{len(rows)}] PSNR u4 {np.mean([x['psnr_uniform4'] for x in per]):.2f} | "
                  f"a22 {np.mean([x['psnr_adaptive22'] for x in per]):.2f} | "
                  f"u2 {np.mean([x['psnr_uniform2'] for x in per]):.2f} | "
                  f"time u4 {np.mean([x['time_uniform4'] for x in per]):.1f}s a22 {np.mean([x['time_adaptive22'] for x in per]):.1f}s")

    with open(os.path.join(run_dir, "per_slice.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader(); w.writerows(per)

    def m(k):
        v = [x[k] for x in per]
        return {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}

    dBA = [x["psnr_adaptive22"] - x["psnr_uniform4"] for x in per]
    dBC = [x["psnr_adaptive22"] - x["psnr_uniform2"] for x in per]
    summary = {
        "n_slices": len(per),
        "psnr_uniform4": m("psnr_uniform4"), "psnr_adaptive22": m("psnr_adaptive22"),
        "psnr_uniform2": m("psnr_uniform2"),
        "ssim_uniform4": m("ssim_uniform4"), "ssim_adaptive22": m("ssim_adaptive22"),
        "time_per_slice_uniform4_s": m("time_uniform4"), "time_per_slice_adaptive22_s": m("time_adaptive22"),
        "time_per_slice_uniform2_s": m("time_uniform2"),
        "delta_psnr_adaptive_vs_uniform4_db": {"mean": round(float(np.mean(dBA)), 4),
                                               "std": round(float(np.std(dBA)), 4),
                                               "win_rate": round(float(np.mean([d > 0 for d in dBA])), 3)},
        "delta_psnr_adaptive_vs_uniform2_db": {"mean": round(float(np.mean(dBC)), 4)},
        "adaptive_extra_chain_rate": round(float(np.mean([x["adaptive_used_extra"] for x in per])), 3),
        "adaptive_recursions_mean": round(float(np.mean([x["adaptive_recursions_mean"] for x in per])), 3),
        "total_min": round((time.time() - t_start) / 60, 1),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 图：三配置 PSNR 逐片对比 + 均值柱状
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = np.arange(len(per))
    axes[0].plot(xs, [x["psnr_uniform4"] for x in per], ".", alpha=0.5, label="uniform N=4")
    axes[0].plot(xs, [x["psnr_adaptive22"] for x in per], ".", alpha=0.5, label="adaptive 2+2")
    axes[0].plot(xs, [x["psnr_uniform2"] for x in per], ".", alpha=0.5, label="uniform N=2")
    axes[0].set_xlabel("切片"); axes[0].set_ylabel("PSNR (dB)"); axes[0].legend(); axes[0].grid(alpha=0.3)
    mus = [summary["psnr_uniform4"]["mean"], summary["psnr_adaptive22"]["mean"], summary["psnr_uniform2"]["mean"]]
    axes[1].bar(["uniform N=4", "adaptive 2+2", "uniform N=2"], mus,
                color=["#94a3b8", "#16a34a", "#cbd5e1"], alpha=0.9)
    for j, v in enumerate(mus):
        axes[1].text(j, v, f"{v:.2f}", ha="center", va="bottom")
    axes[1].set_ylabel("平均 PSNR (dB)"); axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "figures", "adaptive_compare.png"), dpi=150)
    print(f"结果目录 -> {run_dir}")


if __name__ == "__main__":
    main()
