"""BRATS(FeTS21 FLAIR+肿瘤标注) —— 信任图 vs 肿瘤区域分析。

背景：本地 BRATS 官方权重(brats_flair_t2)在、但无多模态配对 GT（FeTS21 切片包仅 FLAIR+标注）。
本脚本不做 PSNR，回答的是创新点的核心问题：
    模型在"合成最不可靠"的病理区域，信任图是否真的掉下来？

指标（均限定在脑区内）：
  A. AUROC(1-trust vs 肿瘤标注)      主指标：低信任像素落在肿瘤区的判别力
  B. 对照基线：FLAIR 亮度、Sobel 梯度幅值（排除"信任图只是在说这里亮/这里难"）
  C. 肿瘤区内 vs 区外的平均信任差（效应量）

输出: results/<date>_brats_flair2t2_trust-tumor/{config.json,per_slice.csv,summary.json,figures/*.png}
用法: python scripts/eval_brats_trust.py [--per-subject 15] [--n-ens 4] [--smoke]
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

from scipy.ndimage import sobel                       # noqa: E402
from trustbridge.engine import BridgeEngine           # noqa: E402
from trustbridge.pipeline import translate_single     # noqa: E402


def auroc(score: np.ndarray, pos: np.ndarray):
    """Mann-Whitney AUROC：score 越高越可能为正类。pos 为 bool。"""
    s = score.ravel().astype(np.float64)
    p = pos.ravel()
    npos = int(p.sum())
    nneg = s.size - npos
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1)
    return float((ranks[p].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="BRATS_FLAIR->T2")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--per-subject", type=int, default=15, help="每受试者取肿瘤切片数")
    ap.add_argument("--min-tumor-frac", type=float, default=0.01)
    ap.add_argument("--n-ens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--smoke", action="store_true", help="冒烟：只跑 5 片")
    args = ap.parse_args()

    man = os.path.join(ROOT, "data", "brats_fets_slices", "manifest.csv")
    rows = [r for r in csv.DictReader(open(man, encoding="utf-8"))
            if float(r["tumor_frac"]) >= args.min_tumor_frac]
    by_subj = {}
    for r in rows:
        by_subj.setdefault(r["subject"], []).append(r)
    sel = []
    for subj in sorted(by_subj):
        rs = sorted(by_subj[subj], key=lambda r: -float(r["tumor_frac"]))
        sel.extend(rs[: args.per_subject])
    if args.smoke:
        sel = sel[:5]
    print(f"评估 {len(sel)} 片 | {args.task} | N={args.n_ens}")

    run_dir = os.path.join(ROOT, "results",
                           f"{datetime.now():%Y-%m-%d}_brats_flair2t2_trust-tumor"
                           + ("_smoke" if args.smoke else ""))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)

    ckpt_path = args.ckpt or str(os.path.join(ROOT, "checkpoints", "brats_flair_t2.ckpt"))
    eng = BridgeEngine(args.task, ckpt_path=ckpt_path)
    print(f"模型: image_size={eng.image_size}, n_steps={eng.n_steps}, "
          f"max_recursions={eng.max_recursions}, 参数量={eng.param_count_m:.1f}M")

    cfg = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "task": args.task, "ckpt": os.path.basename(ckpt_path),
        "ckpt_sha256": file_sha256(ckpt_path), "git_commit": git_commit(),
        "n_ensemble": args.n_ens, "seed": args.seed,
        "data": "data/brats_fets_slices (FeTS21 sliced, 受试者 329-340)",
        "min_tumor_frac": args.min_tumor_frac, "per_subject": args.per_subject,
        "n_slices": len(sel), "smoke": args.smoke,
        "note": "无配对GT: 仅信任-肿瘤标注分析, 无 PSNR/SSIM",
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    per_rows = []
    panels = []          # 供选最佳病例出图
    t_start = time.time()
    for i, r in enumerate(sel):
        case = np.load(r["case_path"]).astype(np.float32)
        label = np.load(r["label_path"])
        brain = case > 0.02
        tumor = (label > 0) & brain

        out = translate_single(eng, case, n_ensemble=args.n_ens,
                               step_skip=1, seed=args.seed)
        pred = out["pred01"]
        res = out["result"]
        trust = res.trust_map
        lowtrust = 1.0 - trust

        grad = np.hypot(sobel(case, axis=0), sobel(case, axis=1))
        a_trust = auroc(lowtrust[brain], tumor[brain])
        a_rcrf = auroc(res.rcrf_map[brain], tumor[brain])
        a_unc = auroc(res.unc_map[brain], tumor[brain])
        a_inten = auroc(case[brain], tumor[brain])
        a_grad = auroc(grad[brain], tumor[brain])
        t_in = float(trust[tumor].mean())
        t_out = float(trust[brain & ~tumor].mean())

        per_rows.append({
            "subject": r["subject"], "z": r["z"], "tumor_frac": r["tumor_frac"],
            "auroc_lowtrust_vs_tumor": round(a_trust, 4),
            "auroc_rcrf_vs_tumor": round(a_rcrf, 4),
            "auroc_unc_vs_tumor": round(a_unc, 4),
            "auroc_intensity_vs_tumor": round(a_inten, 4),
            "auroc_gradient_vs_tumor": round(a_grad, 4),
            "trust_in_tumor": round(t_in, 4), "trust_outside_tumor": round(t_out, 4),
            "trust_score_img": round(res.trust_score, 2), "qc": res.qc_flag,
            "elapsed_s": round(res.elapsed_s, 1),
        })
        panels.append((a_trust, case, pred, trust, tumor, r["subject"], r["z"]))
        if (i + 1) % 5 == 0 or i == len(sel) - 1:
            ms = [x["auroc_lowtrust_vs_tumor"] for x in per_rows]
            mu = [x["auroc_unc_vs_tumor"] for x in per_rows]
            mr = [x["auroc_rcrf_vs_tumor"] for x in per_rows]
            print(f"  [{i+1}/{len(sel)}] AUROC fused {np.mean(ms):.3f} / "
                  f"rcrf {np.mean(mr):.3f} / unc {np.mean(mu):.3f} | {res.elapsed_s:.0f}s/片")

    with open(os.path.join(run_dir, "per_slice.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        w.writeheader()
        w.writerows(per_rows)

    def _mean(k):
        return round(float(np.mean([r[k] for r in per_rows])), 4)

    summary = {
        "n_slices": len(per_rows),
        "auroc_lowtrust_vs_tumor": {"mean": _mean("auroc_lowtrust_vs_tumor"),
                                    "std": round(float(np.std([r["auroc_lowtrust_vs_tumor"] for r in per_rows])), 4)},
        "auroc_intensity_vs_tumor": {"mean": _mean("auroc_intensity_vs_tumor")},
        "auroc_gradient_vs_tumor": {"mean": _mean("auroc_gradient_vs_tumor")},
        "trust_in_tumor": _mean("trust_in_tumor"),
        "trust_outside_tumor": _mean("trust_outside_tumor"),
        "mean_trust_score_img": _mean("trust_score_img"),
        "qc_counts": {f: sum(1 for r in per_rows if r["qc"] == f) for f in ("PASS", "REVIEW", "REJECT")},
        "total_elapsed_min": round((time.time() - t_start) / 60, 1),
        "sec_per_slice": round((time.time() - t_start) / len(per_rows), 1),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # ---- 最佳病例图（AUROC 最高的 6 片）----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    panels.sort(key=lambda x: -x[0])
    for rank, (a, case, pred, trust, tumor, subj, z) in enumerate(panels[:6]):
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
        axes[0].imshow(case, cmap="gray"); axes[0].set_title("输入 FLAIR")
        axes[1].imshow(pred, cmap="gray"); axes[1].set_title("合成 T2")
        im = axes[2].imshow(trust, cmap="RdYlGn", vmin=0, vmax=1)
        axes[2].set_title(f"信任图 (AUROC={a:.3f})")
        fig.colorbar(im, ax=axes[2], fraction=0.046)
        axes[3].imshow(case, cmap="gray")
        axes[3].contour(tumor, colors="red", linewidths=1.2)
        axes[3].set_title("肿瘤标注轮廓")
        for a_ in axes:
            a_.axis("off")
        fig.suptitle(f"{subj} z={z}", y=0.98)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "figures", f"case{rank+1}_{subj}_z{z}.png"), dpi=130)
        plt.close(fig)
    print(f"结果目录 -> {run_dir}")


if __name__ == "__main__":
    main()
