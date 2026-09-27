"""图像级 QC 多特征小模型（limitation ② 的解法）。

背景：图像级 Trust Score（信任图均值）与 PSNR 的秩相关仅 0.37/0.49，
masked 简单聚合已试失败（另一智能体，负结果）。本实验：
  1. 提取 ~16 个推理期可得（无 GT）的图像级特征：信任图分位/体区统计/低信任面积/
     RCRF 与集成分歧聚合/递归收敛统计/源纹理等；
  2. 双方向池化训练（方向哑变量），Ridge 回归预测图像级 PSNR；
  3. 受试者级留一交叉验证（val 6+6 受试者）选特征子集与 alpha，防过拟合；
  4. 测试集评估：Spearman(预测, PSNR) 对比基线 trust_score（0.37/0.49）。

用法:
  python scripts/image_qc_model.py extract --n 120   # 双方向 val+test 一次跑完
  python scripts/image_qc_model.py train
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

OUT_DIR = os.path.join(ROOT, "data", "qc_features")
FEAT_COLS = ["trust_mean", "trust_p10", "trust_p25", "trust_p75",
             "body_trust_mean", "body_trust_p25",
             "low_trust_frac_06", "low_trust_frac_04",
             "rcrf_mean", "rcrf_p90", "unc_mean", "unc_p90",
             "rec_steps_mean", "resid_slope", "ens_disagree_mean",
             "src_grad_mean", "pred_p99", "is_t2src"]


def extract(args):
    import torch  # noqa
    from trustbridge.engine import BridgeEngine  # noqa
    from trustbridge.pipeline import translate_single  # noqa

    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_v3.csv"), encoding="utf-8")))
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "slices.csv")
    new = not os.path.exists(out_csv)
    jobs = []
    for task, ckpt, is_t2src in (("T2->T1", "checkpoints/self_t2t1.ckpt", 0),
                                 ("T1->T2", "checkpoints/self_t1t2.ckpt", 1)):
        man_dir = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                        "manifest_v3.csv"), encoding="utf-8")))
        for split in ("val", "test"):
            rows = [r for r in man_dir if r["split"] == split]
            step = max(1, len(rows) // args.n)
            jobs += [(task, ckpt, split, r, is_t2src) for r in rows[::step][: args.n]]

    done = set()
    if not new:
        for r in csv.DictReader(open(out_csv, encoding="utf-8")):
            done.add((r["task"], r["split"], r["subject"], r["z"]))
    jobs = [j for j in jobs if (j[0], j[2], j[3]["subject"], j[3]["z"]) not in done]
    print(f"待提取 {len(jobs)} 片（已存在 {len(done)} 片跳过）")

    f = open(out_csv, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["task", "split", "subject", "z", "psnr"] + FEAT_COLS)
    eng_cache = {}
    t0 = time.time()
    for i, (task, ckpt, split, r, is_t2src) in enumerate(jobs):
        if task not in eng_cache:
            eng_cache[task] = BridgeEngine(task, ckpt_path=ckpt)
        eng = eng_cache[task]
        src = np.load(r["t2_path"] if task == "T2->T1" else r["t1_path"]).astype(np.float32)
        gt = np.load(r["t1_path"] if task == "T2->T1" else r["t2_path"]).astype(np.float32)
        out = translate_single(eng, src, n_ensemble=args.n_ens, step_skip=1, seed=args.seed)
        res, pred = out["result"], out["pred01"]
        from trustbridge.metrics import mean_norm  # noqa
        g, p = mean_norm(gt), mean_norm(pred)
        from skimage.metrics import peak_signal_noise_ratio  # noqa
        psnr = peak_signal_noise_ratio(g, p, data_range=g.max())

        trust = res.trust_map
        body = pred > 0.02
        tb = trust[body] if body.sum() > 100 else trust.ravel()
        curve = res.step_residuals[-1] if res.step_residuals else []
        slope = (curve[-1] - curve[0]) / max(len(curve), 1) if len(curve) >= 2 else 0.0
        rec = res.recursions_used[-1] if res.recursions_used else [2]
        g_src = np.hypot(np.gradient(src, axis=0), np.gradient(src, axis=1))

        feats = [float(trust.mean()), float(np.percentile(trust, 10)),
                 float(np.percentile(trust, 25)), float(np.percentile(trust, 75)),
                 float(tb.mean()), float(np.percentile(tb, 25)),
                 float((trust[body] < 0.6).mean()) if body.sum() > 100 else 0.0,
                 float((trust[body] < 0.4).mean()) if body.sum() > 100 else 0.0,
                 float(res.rcrf_map.mean()), float(np.percentile(res.rcrf_map, 90)),
                 float(res.unc_map.mean()), float(np.percentile(res.unc_map, 90)),
                 float(np.mean(rec)), float(slope),
                 float(res.members.std(dim=0).mean()),
                 float(g_src.mean()), float(np.percentile(pred, 99)),
                 float(is_t2src)]
        w.writerow([task, split, r["subject"], r["z"], round(float(psnr), 3)] +
                   [round(v, 6) for v in feats])
        if (i + 1) % 20 == 0 or i == len(jobs) - 1:
            f.flush()
            print(f"  [{i+1}/{len(jobs)}] {task} {split} {r['subject']} z={r['z']} "
                  f"({(time.time()-t0)/ (i+1):.1f}s/片)", flush=True)
    f.close()
    print("->", out_csv)


def train(args):
    rows = list(csv.DictReader(open(os.path.join(OUT_DIR, "slices.csv"), encoding="utf-8")))
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    print(f"val {len(val)} 片 / test {len(test)} 片")

    def arr(rs):
        X = np.array([[float(r[c]) for c in FEAT_COLS] for r in rs])
        y = np.array([float(r["psnr"]) for r in rs])
        subj = np.array([r["subject"] for r in rs])
        d = np.array([float(r["is_t2src"]) for r in rs])
        ts = np.array([float(r["trust_mean"]) for r in rs])
        return X, y, subj, d, ts

    Xv, yv, sv, dv, tv = arr(val)
    Xt, yt, st, dt, tt = arr(test)

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    # 候选特征：按 val 单变量 |Spearman| 排序（池化双方向）
    from scipy.stats import spearmanr
    cand = sorted(range(len(FEAT_COLS)),
                  key=lambda j: -abs(spearmanr(Xv[:, j], yv).statistic))

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fit_pred(cols, alpha, Xtr, ytr, Xte):
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        m.fit(Xtr[:, cols], ytr)
        return m.predict(Xte[:, cols])

    # 受试者级留一 CV（val 内，双方向池化）
    subjects = sorted(set(sv))
    best = None
    for k in (5, 8, 12, len(FEAT_COLS)):
        cols = cand[:k]
        for alpha in (0.1, 1.0, 10.0):
            rhos = []
            for s in subjects:
                tr = sv != s
                te = sv == s
                try:
                    pred = fit_pred(cols, alpha, Xv[tr], yv[tr], Xv[te])
                    rhos.append(spearmanr(pred, yv[te]).statistic)
                except Exception:
                    rhos.append(0.0)
            mean_rho = float(np.mean(rhos))
            if best is None or mean_rho > best[0]:
                best = (mean_rho, k, alpha)
    _, k_best, alpha_best = best
    print(f"CV 最优: top-{k_best} 特征, alpha={alpha_best}, "
          f"val 组内平均 Spearman {best[0]:.3f}")

    cols = cand[:k_best]
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha_best))
    model.fit(Xv[:, cols], yv)

    # 测试集评估：分方向 + 池化
    pred_t = model.predict(Xt[:, cols])
    res = {"n_val": len(val), "n_test": len(test), "k": k_best, "alpha": alpha_best,
           "val_cv_spearman": round(best[0], 4), "features": [FEAT_COLS[c] for c in cols],
           "test_spearman": {}, "baseline_trust_spearman": {}}
    for name, mask in (("T2->T1", dt == 0), ("T1->T2", dt == 1), ("pooled", np.ones(len(dt), bool))):
        rho_model = float(spearmanr(pred_t[mask], yt[mask]).statistic)
        rho_base = float(spearmanr(tt[mask], yt[mask]).statistic)
        res["test_spearman"][name] = round(rho_model, 4)
        res["baseline_trust_spearman"][name] = round(rho_base, 4)
        print(f"  {name}: 模型 {rho_model:.3f} vs 基线 {rho_base:.3f}")

    run_dir = os.path.join(ROOT, "results",
                           f"{datetime.now():%Y-%m-%d}_image_qc_model")
    os.makedirs(run_dir, exist_ok=True)
    json.dump(res, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump({"coef_order": [FEAT_COLS[c] for c in cols],
               "coef": list(map(float, model[1].coef_)),
               "intercept": float(model[1].intercept_),
               "scaler_mean": list(map(float, model[0].mean_)),
               "scaler_scale": list(map(float, model[0].scale_))},
              open(os.path.join(run_dir, "model.json"), "w", encoding="utf-8"), indent=2)
    print("->", run_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract", "train"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--n-ens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    if args.phase == "extract":
        extract(args)
    else:
        train(args)
