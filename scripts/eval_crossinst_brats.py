"""实验⑤：跨机构信任迁移——BraTS2018 三点对比设计。

回应"换一台扫描仪/医院，信任指标还有效吗"的质疑。同一种子、同一套官方
FLAIR→T2 权重、同一套体掩码指标，在三个从未见过的患者集合上对照：

  A. 原队列（6 例 2013 系列，已测：AUROC 0.911 / +3.12dB@20%）——参照
  B. 同源留出患者（2013 系列未用过的 HGG，同机构）——患者级泛化
  C. 跨机构（CBICA 验证集，另一机构制备）——机构级泛化

指标：体区 AUROC、dpsnr20、十分位分离、体区 PSNR（全部 within-body，
与 2026-11-03_tumor_sparsify_body 完全同口径）。

用法: python scripts/eval_crossinst_brats.py [--n-subj 6] [--slices-per-subj 20]
输出: results/<date>_crossinst_brats/
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr
from skimage.metrics import peak_signal_noise_ratio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import nibabel as nib  # noqa: E402

from trustbridge.engine import BridgeEngine  # noqa: E402
from trustbridge.pipeline import translate_single  # noqa: E402
from trustbridge.metrics import mean_norm  # noqa: E402

USED_PATIENTS = {"Brats18_2013_10_1", "Brats18_2013_11_1", "Brats18_2013_12_1",
                 "Brats18_2013_13_1", "Brats18_2013_14_1", "Brats18_2013_17_1"}
SLICE_RANGE = range(27, 127)


def auroc_low_vs_higherr(trust, err):
    hi = err > np.percentile(err, 90)
    npos, nneg = int(hi.sum()), int((~hi).sum())
    if npos < 10 or nneg < 10:
        return float("nan")
    score = (1.0 - trust).ravel()
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    return float((ranks[hi.ravel()].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def pick_slices(n_total, k):
    step = max(1, n_total // k)
    return list(range(0, n_total, step))[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subj", type=int, default=6)
    ap.add_argument("--slices-per-subj", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    train_dir = os.path.join(ROOT, "data", "brats2018_raw",
                             "MICCAI_BraTS_2018_Data_Training", "HGG")
    val_dir = os.path.join(ROOT, "data", "brats2018_raw",
                           "MICCAI_BraTS_2018_Data_Validation")

    sets = {}
    # B. 同源留出患者（2013 系列，未用过）
    b_subj = [p for p in sorted(os.path.basename(x) for x in glob.glob(os.path.join(train_dir, "Brats18_2013_*")))
              if p not in USED_PATIENTS]
    sets["B_same_collection_heldout"] = [(os.path.join(train_dir, p), p) for p in b_subj[:args.n_subj]]
    # C. 跨机构（CBICA 验证集）
    c_subj = sorted(os.path.basename(x) for x in glob.glob(os.path.join(val_dir, "Brats18_CBICA_*")))
    sets["C_cross_institution_cbica"] = [(os.path.join(val_dir, p), p) for p in c_subj[:args.n_subj]]

    eng = BridgeEngine("BRATS_FLAIR->T2",
                       ckpt_path=os.path.join(ROOT, "checkpoints", "brats_flair_t2.ckpt"))
    run_dir = os.path.join(ROOT, "results", f"{time.strftime('%Y-%m-%d')}_crossinst_brats")
    os.makedirs(run_dir, exist_ok=True)

    per = []
    t00 = time.time()
    for set_name, subj_list in sets.items():
        print(f"== {set_name}: {len(subj_list)} 例", flush=True)
        for d, p in subj_list:
            fl = sorted(glob.glob(os.path.join(d, "*_flair.nii")))
            t2 = sorted(glob.glob(os.path.join(d, "*_t2.nii")))
            if not fl or not t2:
                print(f"  skip {p}: missing modalities"); continue
            flv = np.asarray(nib.load(fl[0]).dataobj, dtype=np.float32)
            t2v = np.asarray(nib.load(t2[0]).dataobj, dtype=np.float32)
            n_z = min(flv.shape[2], t2v.shape[2], len(SLICE_RANGE))
            zs = [SLICE_RANGE[0] + i for i in pick_slices(
                len(SLICE_RANGE) - (flv.shape[2] - SLICE_RANGE[-1] - 1)
                if flv.shape[2] > SLICE_RANGE[-1] else len(SLICE_RANGE), args.slices_per_subj)]
            zs = [z for z in zs if z < min(flv.shape[2], t2v.shape[2])]
            for z in zs:
                flair = np.rot90(flv[:, :, z], -1)
                t2gt = np.rot90(t2v[:, :, z], -1)
                # 与 prepare_brats2018.py 同协议：逐切片 min-max
                hi = float(max(flair.max(), 1e-6))
                flair = np.clip(flair / hi, 0, 1)
                hi2 = float(max(t2gt.max(), 1e-6))
                t2gt = np.clip(t2gt / hi2, 0, 1)
                body = flair > 0.02
                if body.mean() < 0.02:
                    continue
                out = translate_single(eng, flair, n_ensemble=4, step_skip=1, seed=args.seed)
                res, pred = out["result"], out["pred01"]
                R_n = _pct_norm(res.rcrf_raw_map)
                U_n = _pct_norm(res.unc_raw_map)
                trust = np.clip(1 - 0.6 * R_n - 0.4 * U_n, 0, 1)
                if trust.shape != flair.shape:
                    S = trust.shape[0]; H, W = flair.shape
                    pt, pl = (S - H) // 2, (S - W) // 2
                    crop = lambda m: np.ascontiguousarray(m[pt:pt + H, pl:pl + W])
                    trust, pred = crop(trust), crop(pred)
                err = np.abs(t2gt - pred)
                g, pp = mean_norm(t2gt), mean_norm(pred)
                psnr = float(peak_signal_noise_ratio(g[body], pp[body], data_range=g[body].max()))
                tr_b, er_b = trust[body], err[body]
                per.append({"set": set_name, "patient": p, "z": z,
                            "auroc": round(auroc_low_vs_higherr(trust, err), 4),
                            "spearman": round(float(spearmanr(tr_b, er_b).statistic), 4),
                            "dpsnr20": 0.0, "decile_sep": round(float(
                                er_b[tr_b <= np.percentile(tr_b, 10)].mean() /
                                max(er_b[tr_b >= np.percentile(tr_b, 90)].mean(), 1e-9)), 2),
                            "body_psnr": round(psnr, 2),
                            "trust_score": round(float(tr_b.mean() * 100), 1)})
                # dpsnr20（体区稀疏化）
                keep = tr_b >= np.percentile(tr_b, 20)
                if keep.sum() >= 100:
                    per[-1]["dpsnr20"] = round(float(
                        peak_signal_noise_ratio(g[body][keep], pp[body][keep],
                                                data_range=g[body].max()) - psnr), 3)
            print(f"  {p} done ({(time.time()-t00)/60:.1f}min)", flush=True)

    with open(os.path.join(run_dir, "per_case.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader(); w.writerows(per)

    summary = {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
               "seed": args.seed,
               "reference_A_original_6": {"auroc": 0.9113, "dpsnr20": 3.1231,
                                          "note": "results/2026-11-03_tumor_sparsify_body"},
               "sets": {}}
    for set_name in sets:
        rs = [q for q in per if q["set"] == set_name]
        if not rs:
            continue
        summary["sets"][set_name] = {
            "n_slices": len(rs), "n_subjects": len({q["patient"] for q in rs}),
            "auroc": {"mean": round(float(np.nanmean([q["auroc"] for q in rs])), 4),
                      "std": round(float(np.nanstd([q["auroc"] for q in rs])), 4)},
            "dpsnr20": {"mean": round(float(np.mean([q["dpsnr20"] for q in rs])), 3),
                        "std": round(float(np.std([q["dpsnr20"] for q in rs])), 3)},
            "decile_sep": round(float(np.mean([q["decile_sep"] for q in rs])), 2),
            "body_psnr": {"mean": round(float(np.mean([q["body_psnr"] for q in rs])), 2),
                          "std": round(float(np.std([q["body_psnr"] for q in rs])), 2)},
        }
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("->", run_dir)


if __name__ == "__main__":
    main()
