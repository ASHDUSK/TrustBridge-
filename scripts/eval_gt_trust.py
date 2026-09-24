"""通用带 GT 的信任评估 + 7 行消融/对照（manifest 驱动，CT/BRATS/IXI 通用）。

协议与 scripts/validate_trust.py 完全一致（官方 mean_norm 口径 PSNR/SSIM），
并额外给出消融对照表（每行都是"剔除 20% 像素后的 PSNR 增益"，除 AUROC 外）：
  1 random        随机剔除 20%（证伪稀疏化本身）
  2 gradient      按输入梯度幅值最高 20% 剔除
  3 input_diff    按 |src−gt| 最高 20% 剔除（注意：该行用到 GT，属"泄露型"朴素基线）
  4 rcrf_only     按 RCRF 残差场最高 20% 剔除        ← 创新点①单独
  5 unc_only      按集成不确定性最高 20% 剔除        ← 创新点②单独
  6 fused         按信任图最低 20% 剔除（我们的方法）
  7 oracle        按真实误差最高 20% 剔除（上界）

输出: results/<date>_<task>/{config.json,per_slice.csv,summary.json,figures/*.png}
用法: python scripts/eval_gt_trust.py --task CT_T1->CT --manifest data/synthrad_pelvis_slices/manifest.csv \
        --src-col mr_path --gt-col ct_path [--n 60] [--n-ens 4] [--smoke]
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scipy.stats import pearsonr, spearmanr            # noqa: E402
from scipy.ndimage import sobel                        # noqa: E402
from trustbridge.engine import BridgeEngine            # noqa: E402
from trustbridge.pipeline import translate_single      # noqa: E402
from trustbridge.metrics import mean_norm              # noqa: E402
from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # noqa: E402


def auroc(score, pos):
    s = score.ravel().astype(np.float64)
    p = pos.ravel()
    npos, nneg = int(p.sum()), s.size - int(p.sum())
    if npos == 0 or nneg == 0:
        return np.nan
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1)
    return float((ranks[p].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def psnr_of(g: np.ndarray, p: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(g, p, data_range=g.max()))


def drop_gain(g, p, drop_mask) -> float:
    """剔除 drop_mask(True=剔除) 后的 PSNR − 全图 PSNR。"""
    keep = (~drop_mask).reshape(g.shape)
    full = psnr_of(g, p)
    if keep.sum() < 100:
        return np.nan
    return float(psnr_of(g[keep], p[keep]) - full)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--src-col", required=True)
    ap.add_argument("--gt-col", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--n-ens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--split-col", default=None, help="可选：划分列名(如 split)")
    ap.add_argument("--split-val", default=None, help="可选：划分取值(如 test)")
    ap.add_argument("--tag", default=None, help="结果目录名后缀")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(ROOT, args.manifest), encoding="utf-8")))
    if args.split_col:
        rows = [r for r in rows if r[args.split_col] == args.split_val]
    step = max(1, len(rows) // args.n)
    rows = rows[::step][: args.n]
    if args.smoke:
        rows = rows[:5]
    print(f"评估 {len(rows)} 片 | {args.task} | N={args.n_ens}")

    run_dir = os.path.join(ROOT, "results",
                           f"{datetime.now():%Y-%m-%d}_{args.task.replace('->', '2').replace('*', '_self')}"
                           + (f"_{args.tag}" if args.tag else "")
                           + ("_smoke" if args.smoke else ""))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)

    from trustbridge.engine import TASKS
    ckpt_path = args.ckpt or str(os.path.join(ROOT, "checkpoints", TASKS[args.task][3]))
    eng = BridgeEngine(args.task, ckpt_path=ckpt_path)
    print(f"模型: size={eng.image_size} steps={eng.n_steps} rec={eng.max_recursions} params={eng.param_count_m:.1f}M")

    cfg = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "task": args.task, "ckpt": os.path.basename(ckpt_path),
        "ckpt_sha256": file_sha256(ckpt_path), "git_commit": git_commit(),
        "manifest": args.manifest, "src_col": args.src_col, "gt_col": args.gt_col,
        "n_ensemble": args.n_ens, "seed": args.seed, "n_slices": len(rows),
        "smoke": args.smoke, "protocol": "validate_trust.py 同口径(官方mean_norm PSNR/SSIM) + 7行消融",
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    rng = np.random.RandomState(args.seed)
    per = []
    pix_t, pix_e = [], []
    panels = []
    t0 = time.time()
    for i, r in enumerate(rows):
        src = np.load(os.path.join(ROOT, r[args.src_col])).astype(np.float32)
        gt = np.load(os.path.join(ROOT, r[args.gt_col])).astype(np.float32)
        out = translate_single(eng, src, n_ensemble=args.n_ens,
                               step_skip=1, seed=args.seed, gt01=gt)
        pred, res = out["pred01"], out["result"]
        trust, rcrf_m, unc_m = res.trust_map, res.rcrf_map, res.unc_map

        g, p = mean_norm(gt), mean_norm(pred)
        err = np.abs(g - p)
        hi = err > np.percentile(err, 90)

        psnr = psnr_of(g, p)
        ssim = float(structural_similarity(g, p, data_range=g.max()) * 100)
        a_ = g - g.mean(); b_ = p - p.mean()
        ncc = float((a_ * b_).sum() / (np.sqrt((a_ * a_).sum() * (b_ * b_).sum()) + 1e-9))

        grad = np.hypot(sobel(src, axis=0), sobel(src, axis=1))
        k = int(0.2 * err.size)
        flat = np.arange(err.size)
        rowsel = {
            "random": rng.rand(err.size) < 0.2,
            "gradient": np.argsort(grad.ravel())[-k:] if k > 0 else [],
            "input_diff": np.argsort(np.abs(src - gt).ravel())[-k:] if k > 0 else [],
            "rcrf_only": np.argsort(rcrf_m.ravel())[-k:] if k > 0 else [],
            "unc_only": np.argsort(unc_m.ravel())[-k:] if k > 0 else [],
            "fused": np.argsort(trust.ravel())[:k] if k > 0 else [],
            "oracle": np.argsort(err.ravel())[-k:] if k > 0 else [],
        }
        gains = {}
        for name, sel in rowsel.items():
            mask = np.zeros(err.size, dtype=bool)
            if name == "random":
                mask = sel
            else:
                mask[sel] = True
            gains[name] = round(drop_gain(g, p, mask), 3)

        per.append({
            "idx": i, "psnr": round(psnr, 2), "ssim": round(ssim, 2), "ncc": round(ncc, 4),
            "auroc_fused": round(auroc(1 - trust, hi), 4),
            "auroc_rcrf": round(auroc(rcrf_m, hi), 4),
            "auroc_unc": round(auroc(unc_m, hi), 4),
            "auroc_gradient": round(auroc(grad, hi), 4),
            "gain_random": gains["random"], "gain_gradient": gains["gradient"],
            "gain_input_diff": gains["input_diff"], "gain_rcrf_only": gains["rcrf_only"],
            "gain_unc_only": gains["unc_only"], "gain_fused": gains["fused"],
            "gain_oracle": gains["oracle"],
            "trust_score": round(res.trust_score, 2), "qc": res.qc_flag,
            "elapsed_s": round(res.elapsed_s, 1),
        })
        pix_t.append(trust.ravel()); pix_e.append(err.ravel())
        panels.append((psnr, src, gt, pred, err, trust, i))
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            print(f"  [{i+1}/{len(rows)}] PSNR {np.mean([x['psnr'] for x in per]):.2f} | "
                  f"AUROC {np.nanmean([x['auroc_fused'] for x in per]):.3f} | "
                  f"Δfused {np.nanmean([x['gain_fused'] for x in per]):+.2f}dB | "
                  f"{res.elapsed_s:.0f}s/片")

    with open(os.path.join(run_dir, "per_slice.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader(); w.writerows(per)

    T = np.concatenate(pix_t); E = np.concatenate(pix_e)

    def m(col):
        v = [x[col] for x in per]
        return {"mean": round(float(np.nanmean(v)), 4), "std": round(float(np.nanstd(v)), 4)}

    summary = {
        "n_slices": len(per), "task": args.task,
        "psnr": m("psnr"), "ssim": m("ssim"), "ncc": m("ncc"),
        "pixel_spearman_trust_vs_error": round(float(spearmanr(T, E).statistic), 4),
        "pixel_pearson_trust_vs_error": round(float(pearsonr(T, E).statistic), 4),
        "auroc_fused": m("auroc_fused"), "auroc_rcrf": m("auroc_rcrf"),
        "auroc_unc": m("auroc_unc"), "auroc_gradient": m("auroc_gradient"),
        "gain_random_db": m("gain_random"), "gain_gradient_db": m("gain_gradient"),
        "gain_input_diff_db": m("gain_input_diff"), "gain_rcrf_only_db": m("gain_rcrf_only"),
        "gain_unc_only_db": m("gain_unc_only"), "gain_fused_db": m("gain_fused"),
        "gain_oracle_db": m("gain_oracle"),
        "image_level_spearman_score_vs_psnr": round(
            float(spearmanr([x["trust_score"] for x in per], [x["psnr"] for x in per]).statistic), 4),
        "qc_counts": {f: sum(1 for x in per if x["qc"] == f) for f in ("PASS", "REVIEW", "REJECT")},
        "sec_per_slice": round((time.time() - t0) / len(per), 1),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # ---- 图：7 行消融柱状图 + 代表病例面板 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]

    names = ["random", "gradient", "input_diff", "rcrf_only", "unc_only", "fused", "oracle"]
    vals = [summary[f"gain_{n}_db"]["mean"] for n in names]
    colors = ["#94a3b8", "#94a3b8", "#94a3b8", "#2563eb", "#7c3aed", "#16a34a", "#dc2626"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(names, vals, color=colors, alpha=0.9)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("ΔPSNR @ 剔除20% (dB)")
    ax.set_title(f"{args.task} 消融/对照（n={len(per)}, N={args.n_ens}）")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "figures", "ablation_bar.png"), dpi=150)
    plt.close(fig)

    panels.sort(key=lambda x: x[0])
    for rank, (psnr, src, gt, pred, err, trust, idx) in enumerate(panels[:3] + panels[-3:]):
        fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
        for ax_, img, t in zip(axes,
                               [src, gt, pred, err, trust],
                               ["输入", "GT", "合成", f"|误差|", "信任图"]):
            cm = "hot" if t == "|误差|" else ("RdYlGn" if t == "信任图" else "gray")
            im = ax_.imshow(img, cmap=cm)
            ax_.set_title(t); ax_.axis("off")
        fig.suptitle(f"切片#{idx} PSNR={psnr:.2f}dB", y=0.98)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "figures", f"panel_{rank:02d}_idx{idx}.png"), dpi=120)
        plt.close(fig)
    print(f"结果目录 -> {run_dir}")


if __name__ == "__main__":
    main()
