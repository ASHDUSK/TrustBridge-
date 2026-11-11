"""论文 v0.4 补充实验三合一：collapse / routing / semantic。

collapse  —— 15dB 崩盘报警：官方权重（我们数据上 ~15dB）+ 自训权重（~20dB）
             对照，看信任图/低信任占比能否"不依赖 GT 发出全局警报"。
routing  —— CReF 零成本初筛 + TTA 定点复核的成本-精度曲线：
             混合不确定性 = 不确定区域用 TTA 分歧、其余用 CReF，
             扫阈值画 budget-vs-AUROC，验证"极低成本逼近满预算 TTA"。
semantic —— 语义一致性代理信号（源-合成块级互信息 + 梯度方向一致），
             检验能否补上 T2→T1 的分歧盲区（结构信号 ≠ 方差信号）。

用法:
  python scripts/eval_paper_extras.py collapse --n 20
  python scripts/eval_paper_extras.py routing  --n 30
  python scripts/eval_paper_extras.py semantic --n 60
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr
from skimage.metrics import peak_signal_noise_ratio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trustbridge.engine import BridgeEngine, TASKS  # noqa: E402
from trustbridge.pipeline import translate_single, get_sampler  # noqa: E402
from trustbridge.sampler import _pct_norm  # noqa: E402
from trustbridge.metrics import mean_norm  # noqa: E402
from trustbridge.sampler import _pct_norm  # noqa: E402

TTA_VARIANTS = [(lambda a: a, lambda a: a),
                (np.flipud, np.flipud),
                (np.fliplr, np.fliplr),
                (lambda a: np.rot90(a, 1), lambda a: np.rot90(a, -1))]


def load_test_rows(n):
    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_v3.csv"), encoding="utf-8")))
    rows = [r for r in man if r["split"] == "test"]
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def get_src_gt(r, src_mod):
    t1 = np.load(r["t1_path"]).astype(np.float32)
    t2 = np.load(r["t2_path"]).astype(np.float32)
    return (t2, t1) if src_mod == "T2" else (t1, t2)


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


def body_psnr(gt, pred, mask):
    g, p = mean_norm(gt), mean_norm(pred)
    return float(peak_signal_noise_ratio(g[mask], p[mask], data_range=g[mask].max()))


# ---------------------------------------------------------------- collapse
def cmd_collapse(args):
    run_dir = os.path.join(ROOT, "results", f"{time.strftime('%Y-%m-%d')}_collapse_alarm")
    os.makedirs(run_dir, exist_ok=True)
    rows = load_test_rows(args.n)
    out = []
    for label, task_id, ckpt in [
            ("official_15db", "T1->T2", os.path.join(ROOT, "checkpoints", "ixi_t1_t2.ckpt")),
            ("self_20db", "T1->T2*", os.path.join(ROOT, "checkpoints", "self_t1t2.ckpt"))]:
        eng = BridgeEngine(task_id, ckpt_path=ckpt)
        for i, r in enumerate(rows):
            src, gt = get_src_gt(r, "T1")
            mask = src > 0.02
            o = translate_single(eng, src, n_ensemble=4, step_skip=1, seed=args.seed)
            res, pred = o["result"], o["pred01"]
            # 论文口径融合（CReF 0.6 / 集成 0.4）：原始图单次 p99 归一化
            smp = get_sampler(eng)
            R_n = _pct_norm(smp.rcrf_raw_map)
            U_n = _pct_norm(smp.unc_raw_map)
            tr = np.clip(1 - 0.6 * R_n - 0.4 * U_n, 0, 1)
            if tr.shape != src.shape:
                S = tr.shape[0]; H, W = src.shape
                pt, pl = (S - H) // 2, (S - W) // 2
                tr = np.ascontiguousarray(tr[pt:pt + H, pl:pl + W])
            err = np.abs(gt - pred)
            score = float(tr[mask].mean() * 100.0)
            out.append({
                "model": label, "subject": r["subject"], "z": r["z"],
                "body_psnr": round(body_psnr(gt, pred, mask), 2),
                "lowtrust_ratio_pct": round(float((tr[mask] < 0.6).mean() * 100.0), 2),
                "trust_score": round(score, 1),
                "auroc": round(auroc_low_vs_higherr(tr[mask], err[mask]), 4),
                "qc_flag": "PASS" if score >= 75 else ("REVIEW" if score >= 55 else "REJECT"),
            })
            if (i + 1) % 5 == 0:
                print(f"  [{label} {i+1}/{len(rows)}]", flush=True)
    with open(os.path.join(run_dir, "per_case.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    summary = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "task": "T1->T2", "groups": {}}
    for label in ("official_15db", "self_20db"):
        rs = [q for q in out if q["model"] == label]
        summary["groups"][label] = {
            "n": len(rs),
            "body_psnr": round(float(np.mean([q["body_psnr"] for q in rs])), 2),
            "lowtrust_ratio_pct": round(float(np.mean([q["lowtrust_ratio_pct"] for q in rs])), 2),
            "trust_score": round(float(np.mean([q["trust_score"] for q in rs])), 1),
            "qc_flags": {k: sum(1 for q in rs if q["qc_flag"] == k) for k in
                         ("PASS", "REVIEW", "REJECT")},
        }
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("->", run_dir)


# ---------------------------------------------------------------- routing
def cmd_routing(args):
    run_dir = os.path.join(ROOT, "results", f"{time.strftime('%Y-%m-%d')}_routing_cref_tta")
    os.makedirs(run_dir, exist_ok=True)
    eng = BridgeEngine("T1->T2*")   # 自训模型（与 UQ 基线实验同引擎）
    rows = load_test_rows(args.n)
    per = []
    for i, r in enumerate(rows):
        src, gt = get_src_gt(r, "T1")
        mask = src > 0.02
        err = None
        # CReF 单链（预算 10）
        o1 = translate_single(eng, src, n_ensemble=1, step_skip=1, seed=args.seed)
        res1, pred1 = o1["result"], o1["pred01"]
        u_cref = _pct_norm(res1.rcrf_map)
        # TTA 4 增广（预算 40）
        preds = []
        for fwd_t, inv_t in TTA_VARIANTS:
            o = translate_single(eng, np.ascontiguousarray(fwd_t(src)),
                                 n_ensemble=1, step_skip=1, seed=args.seed)
            p = inv_t(np.ascontiguousarray(o["pred01"]))
            if p.shape != src.shape:
                p = p[:src.shape[0], :src.shape[1]]
            preds.append(p)
        u_tta = _pct_norm(np.std(preds, axis=0))
        if err is None:
            err = np.abs(gt - preds[0])
        # 混合：CReF 高于阈值 t 的像素用 TTA 分歧，其余用 CReF
        for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01):
            routed = u_cref >= t
            frac = float(routed[mask].mean())
            hybrid = np.where(routed, u_tta, u_cref)
            per.append({"subject": r["subject"], "z": r["z"], "threshold": t,
                        "budget_fwd": round(10 + 40 * frac, 1),
                        "routed_frac": round(frac, 4),
                        "auroc": round(auroc_low_vs_higherr(1 - hybrid, err), 4)})
        per.append({"subject": r["subject"], "z": r["z"], "threshold": -1,
                    "budget_fwd": 10, "routed_frac": 0.0,
                    "auroc": round(auroc_low_vs_higherr(1 - u_cref, err), 4)})
        per.append({"subject": r["subject"], "z": r["z"], "threshold": 0,
                    "budget_fwd": 50, "routed_frac": 1.0,
                    "auroc": round(auroc_low_vs_higherr(1 - u_tta, err), 4)})
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(rows)}]", flush=True)
    with open(os.path.join(run_dir, "per_case.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader(); w.writerows(per)
    curve = {}
    for t in sorted({q["threshold"] for q in per}):
        rs = [q for q in per if q["threshold"] == t]
        curve[str(t)] = {"budget": round(float(np.mean([q["budget_fwd"] for q in rs])), 1),
                         "auroc": round(float(np.nanmean([q["auroc"] for q in rs])), 4),
                         "routed_frac": round(float(np.mean([q["routed_frac"] for q in rs])), 3)}
    summary = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "task": "T1->T2*",
               "note": "混合不确定性：CReF>=t 区域用 TTA 分歧，其余用 CReF；t=-1 纯CReF，t=0 纯TTA",
               "cost_auroc_curve": curve}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(curve, ensure_ascii=False, indent=1))
    print("->", run_dir)


# ---------------------------------------------------------------- semantic
def _block_mi(src, pred, block=16):
    """源-合成块级归一化互信息 → 逐块一致性（0~1，越大越一致）。"""
    H, W = src.shape
    mi = np.zeros((H // block, W // block), np.float32)
    nb = 256 // block
    for bi in range(min(nb, H // block)):
        for bj in range(min(nb, W // block)):
            a = src[bi * block:(bi + 1) * block, bj * block:(bj + 1) * block].ravel()
            b = pred[bi * block:(bi + 1) * block, bj * block:(bj + 1) * block].ravel()
            h, _, _ = np.histogram2d(a, b, bins=32)
            pxy = h / max(h.sum(), 1.0)
            px, py = pxy.sum(1), pxy.sum(0)
            nz = pxy > 0
            mi_xy = float((pxy[nz] * np.log2(pxy[nz] /
                          np.outer(px, py)[nz])).sum())
            hx = -float(np.sum(px[px > 0] * np.log2(px[px > 0])))
            hy = -float(np.sum(py[py > 0] * np.log2(py[py > 0])))
            denom = max(hx + hy, 1e-6)
            mi[bi, bj] = np.clip(2 * mi_xy / denom, 0, 1)   # 归一化 MI ∈ [0,1]
    return mi


def _grad_agree(src, pred, mask):
    """梯度方向一致性：源与合成在共享解剖结构上的边缘方向余弦。"""
    gy1, gx1 = np.gradient(src)
    gy2, gx2 = np.gradient(pred)
    n1 = np.hypot(gx1, gy1) + 1e-8
    n2 = np.hypot(gx2, gy2) + 1e-8
    cos = (gx1 * gx2 + gy1 * gy2) / (n1 * n2)
    w = np.minimum(n1, n2)                      # 两侧都有边缘处权重才大
    agree = (cos * w)
    # 平滑到像素级：把强边缘处的方向一致度弱扩散
    from scipy.ndimage import gaussian_filter
    m = np.where(w > np.percentile(w, 80), agree, 0.0)
    sm = gaussian_filter(m, sigma=8)
    norm = gaussian_filter(np.where(w > np.percentile(w, 80), 1.0, 0.0), sigma=8) + 1e-6
    return (sm / norm).astype(np.float32)


def cmd_semantic(args):
    run_dir = os.path.join(ROOT, "results", f"{time.strftime('%Y-%m-%d')}_semantic_proxy")
    os.makedirs(run_dir, exist_ok=True)
    out = []
    for task_id in ("T1->T2*", "T2->T1*"):
        eng = BridgeEngine(task_id)
        _, src_mod, _, _ = TASKS[task_id]
        rows = load_test_rows(args.n // 2)
        for i, r in enumerate(rows):
            src, gt = get_src_gt(r, src_mod)
            mask = src > 0.02
            o = translate_single(eng, src, n_ensemble=4, step_skip=1, seed=args.seed)
            res, pred = o["result"], o["pred01"]
            if res.trust_map.shape != src.shape:
                S = res.trust_map.shape[0]; H, W = src.shape
                pt, pl = (S - H) // 2, (S - W) // 2
                res.trust_map = np.ascontiguousarray(res.trust_map[pt:pt + H, pl:pl + W])
                res.rcrf_map = np.ascontiguousarray(res.rcrf_map[pt:pt + H, pl:pl + W])
                res.unc_map = np.ascontiguousarray(res.unc_map[pt:pt + H, pl:pl + W])
                pred = np.ascontiguousarray(pred[pt:pt + H, pl:pl + W])
            err = np.abs(gt - pred)
            nmi = _block_mi(src, pred)
            nmi_up = np.kron(nmi, np.ones((16, 16), np.float32))[:src.shape[0], :src.shape[1]]
            ga = _grad_agree(src, pred, mask)
            fused_struct = 1 - _pct_norm(0.5 * (1 - nmi_up) + 0.5 * (1 - ga))  # 不一致→低信任
            disp = 1 - (0.6 * _pct_norm(res.rcrf_map) + 0.4 * _pct_norm(res.unc_map))
            combo = 1 - _pct_norm(0.5 * (1 - fused_struct) + 0.5 * (1 - disp))
            for name, sig in (("nmi", 1 - nmi_up), ("grad_agree", 1 - ga),
                              ("struct_fused", 1 - fused_struct),
                              ("dispersion_ours", 1 - disp),
                              ("struct+disp", 1 - combo)):
                out.append({"task": task_id, "subject": r["subject"], "z": r["z"],
                            "signal": name,
                            "auroc": round(auroc_low_vs_higherr(sig, err), 4),
                            "spearman": round(float(spearmanr(
                                sig[mask], err[mask]).statistic), 4)})
            if (i + 1) % 10 == 0:
                print(f"  [{task_id} {i+1}/{len(rows)}]", flush=True)
    with open(os.path.join(run_dir, "per_case.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    summary = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "groups": {}}
    for task_id in ("T1->T2*", "T2->T1*"):
        for sig in ("nmi", "grad_agree", "struct_fused", "dispersion_ours", "struct+disp"):
            rs = [q for q in out if q["task"] == task_id and q["signal"] == sig]
            if rs:
                summary["groups"][f"{task_id}:{sig}"] = {
                    "n": len(rs),
                    "auroc": round(float(np.nanmean([q["auroc"] for q in rs])), 4),
                    "spearman": round(float(np.nanmean([q["spearman"] for q in rs])), 4)}
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("->", run_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("collapse", "routing", "semantic"):
        sp = sub.add_parser(name)
        sp.add_argument("--n", type=int, default=30)
        sp.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    {"collapse": cmd_collapse, "routing": cmd_routing,
     "semantic": cmd_semantic}[args.cmd](args)
