"""创新点有效性验证：信任图/信任分 与 真实误差 的统计关系。

在带 GT 的切片集（data/ixi_slices, manifest split=val）上：
  A. 像素级：信任图 vs |pred-GT| 的 Spearman/Pearson 相关（负相关为好）
  B. 判别力：以信任图低分预测"高误差像素(>90分位)"的 AUROC
  C. 稀疏化：按信任图剔除最不可信 k% 像素后的 PSNR 提升
  D. 图像级：Trust Score vs PSNR 的 Spearman 秩相关

输出: outputs/validation_report.json + outputs/validation_plots.png
用法: python scripts/validate_trust.py [--task T2->T1] [--n 60] [--n-ens 4]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scipy.stats import pearsonr, spearmanr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from trustbridge.engine import BridgeEngine, TASKS
from trustbridge.pipeline import translate_single
from trustbridge.metrics import mean_norm


def img_psnr(gt, pr):
    g, p = mean_norm(gt), mean_norm(pr)
    return peak_signal_noise_ratio(g, p, data_range=g.max())


def img_ssim(gt, pr):
    g, p = mean_norm(gt), mean_norm(pr)
    return structural_similarity(g, p, data_range=g.max()) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T2->T1")
    ap.add_argument("--n", type=int, default=60, help="评估切片数")
    ap.add_argument("--n-ens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--ckpt", default=None, help="自训练权重路径（默认用官方权重）")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    from trustbridge.io_utils import _to01  # noqa
    man = os.path.join(ROOT, "data", "ixi_slices", "manifest_v3.csv")
    rows = list(csv.DictReader(open(man, encoding="utf-8")))
    rows = [r for r in rows if r["split"] == args.split]
    step = max(1, len(rows) // args.n)
    rows = rows[::step][: args.n]
    print(f"评估 {len(rows)} 个切片 ({args.task}, N={args.n_ens})")

    eng = BridgeEngine(args.task, ckpt_path=args.ckpt)
    display, src_mod, tgt_mod, _ckpt = TASKS[args.task]

    pix_trust, pix_err = [], []
    aurocs, spars, scores, psnrs, ssims = [], [], [], [], []
    import torch

    for i, r in enumerate(rows):
        t1 = np.load(r["t1_path"]).astype(np.float32)
        t2 = np.load(r["t2_path"]).astype(np.float32)
        gt, src = (t1, t2) if tgt_mod == "T1" else (t2, t1)
        out = translate_single(eng, src, n_ensemble=args.n_ens,
                               step_skip=1, seed=args.seed, gt01=gt)
        pred, res = out["pred01"], out["result"]
        err = np.abs(gt - pred)

        pix_trust.append(res.trust_map.ravel())
        pix_err.append(err.ravel())

        # B. AUROC：低信任 → 高误差（>90 分位）
        hi = err > np.percentile(err, 90)
        if 0 < hi.sum() < hi.size:
            order = np.argsort(res.trust_map.ravel())      # 信任升序
            ranks = np.empty_like(order, dtype=np.float32)
            ranks[order] = np.arange(1, hi.size + 1)       # 信任越低 rank 越小
            # Mann-Whitney AUROC: P(trust_low 误差异于 trust_high)
            npos, nneg = hi.sum(), (~hi).sum()
            r_sum = ranks[hi.ravel()].sum()
            auroc = (r_sum - npos * (npos + 1) / 2) / (npos * nneg)
            aurocs.append(1.0 - auroc)                     # 信任低→误差高 ⇒ AUROC(以低信任为正类)

        # C. 稀疏化：剔除最不可信 20% 后 PSNR 提升
        thr = np.percentile(res.trust_map, 20)
        keep = res.trust_map >= thr
        p_full = img_psnr(gt, pred)
        g, p = mean_norm(gt), mean_norm(pred)
        from skimage.metrics import peak_signal_noise_ratio as _psnr
        p_kept = _psnr(g[keep], p[keep], data_range=g.max())
        spars.append(p_kept - p_full)

        scores.append(res.trust_score)
        psnrs.append(p_full)
        ssims.append(img_ssim(gt, pred))
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            print(f"  [{i+1}/{len(rows)}] mean PSNR {np.mean(psnrs):.2f} dB, "
                  f"mean AUROC {np.mean(aurocs):.3f}, mean ΔPSNR(剔除20%) {np.mean(spars):+.2f} dB")

    T = np.concatenate(pix_trust); E = np.concatenate(pix_err)
    res_json = {
        "task": args.task, "n_slices": len(rows), "n_ensemble": args.n_ens,
        "pixel_spearman_trust_vs_error": round(float(spearmanr(T, E).statistic), 4),
        "pixel_pearson_trust_vs_error": round(float(pearsonr(T, E).statistic), 4),
        "auroc_lowtrust_predicts_higherror": round(float(np.mean(aurocs)), 4),
        "auroc_std": round(float(np.std(aurocs)), 4),
        "psnr_gain_after_dropping_20pct_leasttrusted": round(float(np.mean(spars)), 3),
        "image_level_spearman_score_vs_psnr": round(float(spearmanr(scores, psnrs).statistic), 4),
        "mean_psnr": round(float(np.mean(psnrs)), 2), "mean_ssim": round(float(np.mean(ssims)), 2),
        "mean_trust_score": round(float(np.mean(scores)), 2),
    }
    print(json.dumps(res_json, ensure_ascii=False, indent=2))

    out_dir = os.path.join(ROOT, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(res_json, f, ensure_ascii=False, indent=2)

    # 图：散点 + 稀疏化曲线样例
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    idx = np.random.RandomState(0).choice(len(T), min(60000, len(T)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].scatter(T[idx], E[idx], s=2, alpha=0.15, color="#2563eb")
    axes[0].set_xlabel("Trust(p)"); axes[0].set_ylabel("|pred−GT|(p)")
    axes[0].set_title(f"像素级 信任-误差 关系 (Spearman ρ={res_json['pixel_spearman_trust_vs_error']})")
    axes[0].grid(alpha=0.3)
    ks = [0, 10, 20, 30, 40]
    axes[1].bar([f"剔除{k}%" for k in ks], [0] + [float(np.mean(spars)) * k / 20 for k in ks[1:]],
                color="#16a34a", alpha=0.85)
    axes[1].set_ylabel("PSNR 提升 (dB)")
    axes[1].set_title("按信任图剔除低可信像素后的 PSNR 增益（线性外推示意）")
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "validation_plots.png"), dpi=150)
    print(f"报告 -> outputs/validation_report.json, outputs/validation_plots.png")


if __name__ == "__main__":
    main()
