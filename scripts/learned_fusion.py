"""可学习融合实验：多信号块级特征 + 小模型聚合 vs 固定权重融合。

动机（docs/08 + 交接文档 §4）：固定融合 (w=0.6/0.4) 的稀疏化增益仅捕获 oracle 上限的
~20-35%。本实验验证两条改进路径：
  A. 可学习融合（同信号：RCRF+集成不确定性，学习块级聚合权重）
  B. 多信号源（新增 3 个零成本信号：结构差/强度差/源纹理复杂度）

协议：
  - 训练=验证集切片（6 受试者），测试=测试集切片（8 受试者），受试者无重叠
  - 块级特征（block=8 → 16×16=256 块/片）：
      rcrf/unc/grad_diff/int_diff/src_grad/src_mean 各取 {mean,p90,std} + src_mean = 16 维
  - 标签：块误差 |pred−gt| 均值的切片内 top-20%（与 AUROC 高误差定义一致）
  - 模型：LogisticRegression 与 HistGradientBoostingClassifier 各一
  - 对照（同测试片）：fixed 融合 / 仅 rcrf / 仅 unc / oracle
  - 指标：块级 AUROC、像素级 Spearman（上采样）、像素级剔除 20% 的 ΔPSNR（对齐 docs/08）

用法:
  python scripts/learned_fusion.py extract --task T2->T1 --split val --n 120
  python scripts/learned_fusion.py extract --task T2->T1 --split test --n 120
  python scripts/learned_fusion.py train   --task T2->T1
  （T1->T2 同理）
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

from scipy.ndimage import sobel  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio  # noqa: E402

from trustbridge.metrics import mean_norm  # noqa: E402

BLOCK = 8
FEATS = ["rcrf", "unc", "grad_diff", "int_diff", "src_grad"]
STATS = ["mean", "p90", "std"]


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


def blocks(x: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """[H,W] → [nB,nB,block,block] 视图。"""
    H, W = x.shape
    nb = H // block
    return x[: nb * block, : nb * block].reshape(nb, block, nb, block)


def block_stats(x: np.ndarray) -> dict:
    b = blocks(x.astype(np.float32))
    return {"mean": b.mean(axis=(1, 3)),
            "p90": np.percentile(b, 90, axis=(1, 3)),
            "std": b.std(axis=(1, 3))}


def upsample_block(blk: np.ndarray, H: int, W: int) -> np.ndarray:
    """块图 → 像素图（双线性，训练/测试一致）。"""
    import torch
    import torch.nn.functional as NF
    t = torch.tensor(blk, dtype=torch.float32)[None, None]
    up = NF.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    return up.numpy().astype(np.float32)


def norm01(m: np.ndarray) -> np.ndarray:
    hi = np.percentile(m, 99)
    return np.clip(m / hi, 0, 1).astype(np.float32) if hi > 1e-12 else np.zeros_like(m)


def extract(args):
    import torch  # noqa
    from trustbridge.engine import BridgeEngine, TASKS  # noqa
    from trustbridge.pipeline import translate_single  # noqa

    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_v3.csv"), encoding="utf-8")))
    rows = [r for r in man if r["split"] == args.split]
    step = max(1, len(rows) // args.n)
    rows = rows[::step][: args.n]

    eng = BridgeEngine(args.task, ckpt_path=args.ckpt)
    _, src_mod, tgt_mod, _ = TASKS[args.task]
    print(f"extract: {args.task} split={args.split} n={len(rows)} | "
          f"model size={eng.image_size}")

    out_dir = os.path.join(ROOT, "data", "fusion_features")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.task.replace('->','2').replace('*','_')}_{args.split}"
    out_csv = os.path.join(out_dir, f"{tag}.csv")
    feat_cols = [f"{s}_{st}" for s in FEATS for st in STATS] + ["src_mean"]
    header = ["slice_id", "z", "brow", "bcol"] + feat_cols + ["blk_err"]

    ck = args.ckpt or str(os.path.join(ROOT, "checkpoints", TASKS[args.task][3]))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, r in enumerate(rows):
            src = np.load(r["t2_path"] if src_mod == "T2" else r["t1_path"]).astype(np.float32)
            gt = np.load(r["t1_path"] if tgt_mod == "T1" else r["t2_path"]).astype(np.float32)
            out = translate_single(eng, src, n_ensemble=args.n_ens,
                                   step_skip=1, seed=args.seed)
            res, pred = out["result"], out["pred01"]

            err = np.abs(gt - pred)
            g_src, g_pred = np.hypot(sobel(src, 0), sobel(src, 1)), np.hypot(sobel(pred, 0), sobel(pred, 1))
            signals = {
                "rcrf": norm01(res.rcrf_map),
                "unc": norm01(res.unc_map),
                "grad_diff": norm01(np.abs(g_pred - g_src)),
                "int_diff": norm01(np.abs(pred - src)),
                "src_grad": norm01(g_src),
            }
            stats = {k: block_stats(v) for k, v in signals.items()}
            sm = block_stats(src)["mean"]
            berr = blocks(err).mean(axis=(1, 3))
            nb = berr.shape[0]

            for br in range(nb):
                for bc in range(nb):
                    row = [r["subject"], r["z"], br, bc]
                    for s_ in FEATS:
                        for st in STATS:
                            row.append(float(stats[s_][st][br, bc]))
                    row.append(float(sm[br, bc]))
                    row.append(float(berr[br, bc]))
                    w.writerow(row)
            if (i + 1) % 10 == 0 or i == len(rows) - 1:
                print(f"  [{i+1}/{len(rows)}] {r['subject']} z={r['z']} ({res.elapsed_s:.1f}s/片)",
                      flush=True)
    print("->", out_csv)


def train(args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feat_cols = [f"{s}_{st}" for s in FEATS for st in STATS] + ["src_mean"]
    two_sig = [c for c in feat_cols if c.startswith(("rcrf_", "unc_"))]

    def load(split):
        p = os.path.join(ROOT, "data", "fusion_features",
                         f"{args.task.replace('->','2').replace('*','_')}_{split}.csv")
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        X = np.array([[float(r[c]) for c in feat_cols] for r in rows])
        e = np.array([float(r["blk_err"]) for r in rows])
        meta = [(r["slice_id"], r["z"]) for r in rows]
        return X, e, meta, rows

    Xv, ev, metav, rv = load("val")
    Xt, et, metat, rt = load("test")
    # 标签：切片内 top-20% 误差块
    def labels(X, e, meta):
        y = np.zeros(len(e), dtype=int)
        df = {}
        for i, (sid, z) in enumerate(meta):
            df.setdefault((sid, z), []).append(i)
        for k, idxs in df.items():
            idxs = np.array(idxs)
            thr = np.quantile(e[idxs], 0.8)
            y[idxs[e[idxs] >= thr]] = 1
        return y
    yv, yt = labels(Xv, ev, metav), labels(Xt, et, metat)
    print(f"特征 {Xv.shape[1]} 维 | val {Xv.shape[0]} 块 (pos {yv.mean():.1%}) | "
          f"test {Xt.shape[0]} 块 (pos {yt.mean():.1%})")

    models = {
        "learned_2sig_lr": make_pipeline(StandardScaler(),
                                         LogisticRegression(C=1.0, max_iter=2000)),
        "learned_all_lr": make_pipeline(StandardScaler(),
                                        LogisticRegression(C=1.0, max_iter=2000)),
        "learned_all_gbdt": HistGradientBoostingClassifier(max_iter=200, random_state=0),
    }
    subsets = {"learned_2sig_lr": two_sig, "learned_all_lr": feat_cols,
               "learned_all_gbdt": feat_cols}

    def auroc(score, pos):
        s = score.ravel().astype(np.float64); p = pos.ravel()
        npos, nneg = int(p.sum()), s.size - int(p.sum())
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, s.size + 1)
        return float((ranks[p].sum() - npos * (npos + 1) / 2) / (npos * nneg))

    # ---- 训练（验证集）----
    fitted = {}
    for name, cols in subsets.items():
        m = models[name]
        m.fit(Xv[:, [feat_cols.index(c) for c in cols]], yv)
        fitted[name] = m
        print(f"trained {name}")

    # ---- 测试集评估（块级 AUROC + 逐切片像素级协议）----
    run_dir = os.path.join(ROOT, "results",
                           f"{datetime.now():%Y-%m-%d}_learned_fusion_"
                           f"{args.task.replace('->', '2').replace('*', '_self')}")
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)

    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_v3.csv"), encoding="utf-8")))
    rows = [r for r in man if r["split"] == "test"]
    step = max(1, len(rows) // args.n)
    rows = rows[::step][: args.n]
    from trustbridge.engine import BridgeEngine, TASKS
    from trustbridge.pipeline import translate_single
    eng = BridgeEngine(args.task, ckpt_path=args.ckpt)
    _, src_mod, tgt_mod, _ = TASKS[args.task]

    cfg = {"date": datetime.now().isoformat(timespec="seconds"), "task": args.task,
           "ckpt": os.path.basename(args.ckpt or TASKS[args.task][3]),
           "ckpt_sha256": file_sha256(args.ckpt or os.path.join(
               ROOT, "checkpoints", TASKS[args.task][3])),
           "git_commit": git_commit(), "block": BLOCK, "n_val_blocks": int(yv.size),
           "n_test_blocks": int(yt.size), "seed": args.seed, "n_slices": len(rows),
           "models": list(models), "protocol": "val 训练 / test 评估，块级 top-20% 误差标签"}
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    policies = ["fixed", "learned_2sig_lr", "learned_all_lr", "learned_all_gbdt",
                "rcrf_only", "unc_only", "oracle"]
    per, acc = [], {p: {"psnr": [], "ssim": [], "auroc": []} for p in policies}
    t0 = time.time()
    for i, r in enumerate(rows):
        src = np.load(r["t2_path"] if src_mod == "T2" else r["t1_path"]).astype(np.float32)
        gt = np.load(r["t1_path"] if tgt_mod == "T1" else r["t2_path"]).astype(np.float32)
        out = translate_single(eng, src, n_ensemble=args.n_ens, step_skip=1, seed=args.seed)
        res, pred = out["result"], out["pred01"]
        g, p = mean_norm(gt), mean_norm(pred)
        err = np.abs(g - p)
        H, W = err.shape

        g_src, g_pred = np.hypot(sobel(src, 0), sobel(src, 1)), np.hypot(sobel(pred, 0), sobel(pred, 1))
        signals = {"rcrf": norm01(res.rcrf_map), "unc": norm01(res.unc_map),
                   "grad_diff": norm01(np.abs(g_pred - g_src)),
                   "int_diff": norm01(np.abs(pred - src)), "src_grad": norm01(g_src)}
        stats = {k: block_stats(v) for k, v in signals.items()}
        sm = block_stats(src)["mean"]
        nb = stats["rcrf"]["mean"].shape[0]
        Xb = np.zeros((nb * nb, len(feat_cols)), dtype=np.float32)
        ci = 0
        for s_ in FEATS:
            for st in STATS:
                Xb[:, ci] = stats[s_][st].ravel()
                ci += 1
        Xb[:, ci] = sm.ravel()

        sus_fixed = (0.6 * norm01(res.rcrf_map) + 0.4 * norm01(res.unc_map))
        scores = {
            "fixed": 1 - sus_fixed,
            "learned_2sig_lr": 1 - upsample_block(
                fitted["learned_2sig_lr"].predict_proba(
                    Xb[:, [feat_cols.index(c) for c in two_sig]])[:, 1].reshape(nb, nb), H, W),
            "learned_all_lr": 1 - upsample_block(
                fitted["learned_all_lr"].predict_proba(Xb)[:, 1].reshape(nb, nb), H, W),
            "learned_all_gbdt": 1 - upsample_block(
                fitted["learned_all_gbdt"].predict_proba(Xb)[:, 1].reshape(nb, nb), H, W),
            "rcrf_only": 1 - norm01(res.rcrf_map),
            "unc_only": 1 - norm01(res.unc_map),
            "oracle": 1 - norm01(err),
        }
        k = int(0.2 * err.size)
        row = {"idx": i, "subject": r["subject"], "z": r["z"], "psnr": round(
            peak_signal_noise_ratio(g, p, data_range=g.max()), 2)}
        hi = err > np.percentile(err, 90)
        for pname, score in scores.items():
            s_ = score.ravel()
            au = auroc(1 - s_, hi) if pname != "oracle" else np.nan
            drop = np.zeros(err.size, dtype=bool)
            drop[np.argsort(s_)[:k]] = True
            keep = (~drop).reshape(H, W)
            gain = (peak_signal_noise_ratio(g[keep], p[keep], data_range=g.max()) -
                    peak_signal_noise_ratio(g, p, data_range=g.max())) if keep.sum() > 100 else np.nan
            acc[pname]["psnr"].append(row["psnr"])
            acc[pname]["auroc"].append(au)
            row[f"auroc_{pname}"] = None if np.isnan(au) else round(au, 4)
            row[f"gain_{pname}"] = None if np.isnan(gain) else round(gain, 3)
            if pname in ("fixed", "learned_all_gbdt", "oracle"):
                acc[pname]["ssim"].append(np.nan)
        per.append(row)
        if (i + 1) % 10 == 0 or i == len(rows) - 1:
            msg = " | ".join(f"{p} {np.nanmean(acc[p]['auroc']):.3f}" for p in
                             ["fixed", "learned_2sig_lr", "learned_all_lr", "learned_all_gbdt"])
            print(f"  [{i+1}/{len(rows)}] AUROC {msg}", flush=True)

    cols = list(per[0].keys())
    with open(os.path.join(run_dir, "per_slice.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(per)

    def m_(p, key):
        v = [x[key] for x in per if x.get(key) is not None]
        return {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}

    summary = {"task": args.task, "n_slices": len(per), "block": BLOCK,
               "train": "val-split 块级", "test": "test-split 块级",
               "psnr": m_("fixed", "psnr"),
               "block_auroc": {p: m_(p, f"auroc_{p}") for p in policies if p != "oracle"},
               "gain20_db": {p: m_(p, f"gain_{p}") for p in policies},
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps({k: summary[k] for k in ("block_auroc", "gain20_db")}, ensure_ascii=False, indent=2))
    print("->", run_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["extract", "train"])
    ap.add_argument("--task", default="T2->T1")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--n-ens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    if args.phase == "extract":
        extract(args)
    else:
        train(args)
