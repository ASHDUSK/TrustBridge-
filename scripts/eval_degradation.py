"""实验④：输入退化剂量-响应——信任图与输入门控能否"知道自己不知道"。

设计（回应外部评审共同建议）：
  对输入施加可控退化（高斯模糊 / 高斯噪声 / 刚性错位）× 剂量梯度，
  每个条件跑完整翻译（N=4），测三条响应曲线：
    1) 翻译质量随剂量恶化（体区 PSNR ↓）——确认退化确实伤害输出
    2) 信任图响应：体区平均信任是否单调下降？AUROC 是否保持定位能力？
    3) 输入门控响应：maha 距离是否随剂量上升（门控能否提前察觉）？

用法: python scripts/eval_degradation.py [--n 30]
输出: results/<date>_degradation_response/
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

from trustbridge.engine import BridgeEngine  # noqa: E402
from trustbridge.pipeline import translate_single, get_sampler  # noqa: E402
from trustbridge.metrics import mean_norm  # noqa: E402
from trustbridge.sampler import _pct_norm  # noqa: E402
from trustbridge.gate_v2 import GateV2  # noqa: E402


def degrade(img, kind, dose):
    if kind == "none":
        return img
    if kind == "blur":
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(img, sigma=dose).astype(np.float32)
    if kind == "noise":
        rng = np.random.RandomState(42)
        return np.clip(img + rng.normal(0, dose, img.shape).astype(np.float32), 0, 1)
    if kind == "shift":
        k = int(dose)
        out = np.zeros_like(img)
        out[:, k:] = img[:, :-k]          # 水平平移（模拟错位）
        return out
    raise ValueError(kind)



def align_map(m, ref):
    """信任图对齐到源尺寸：小于模型尺寸则中心裁剪，大于则双线性放大（同 pipeline）。"""
    S = m.shape[0]; H, W = ref.shape
    if H <= S and W <= S:
        pt, pl = (S - H) // 2, (S - W) // 2
        return np.ascontiguousarray(m[pt:pt + H, pl:pl + W])
    from skimage.transform import resize
    return resize(m, (H, W), order=1, preserve_range=True).astype(np.float32)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T1->T2")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    CONDITIONS = [("none", 0), ("blur", 0.5), ("blur", 1.0), ("blur", 2.0),
                  ("noise", 0.01), ("noise", 0.03), ("noise", 0.06), ("noise", 0.1),
                  ("shift", 2), ("shift", 4), ("shift", 8)]

    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_v3.csv"), encoding="utf-8")))
    rows = [r for r in man if r["split"] == "test"]
    step = max(1, len(rows) // args.n)
    rows = rows[::step][: args.n]

    eng = BridgeEngine(args.task, ckpt_path=args.ckpt)
    gate = GateV2()
    run_dir = os.path.join(ROOT, "results", f"{time.strftime('%Y-%m-%d')}_degradation_response")
    os.makedirs(run_dir, exist_ok=True)

    per = []
    t00 = time.time()
    for i, r in enumerate(rows):
        t1 = np.load(r["t1_path"]).astype(np.float32)
        t2 = np.load(r["t2_path"]).astype(np.float32)
        src, gt = (t2, t1) if "T2->" in args.task else (t1, t2)
        mask = src > 0.02
        for kind, dose in CONDITIONS:
            x = degrade(src, kind, dose)
            out = translate_single(eng, x, n_ensemble=4, step_skip=1, seed=args.seed)
            res, pred = out["result"], out["pred01"]
            # 论文口径融合（CReF 0.6 / 集成 0.4），从原始图重算，不依赖 sampler 默认
            smp = get_sampler(eng)
            R_n = _pct_norm(smp.rcrf_raw_map)
            U_n = _pct_norm(smp.unc_raw_map)
            tr = np.clip(1 - 0.6 * R_n - 0.4 * U_n, 0, 1)
            tr = align_map(tr, x)
            err = np.abs(gt - pred)
            g, p = mean_norm(gt), mean_norm(pred)
            psnr = float(peak_signal_noise_ratio(g[mask], p[mask], data_range=g[mask].max()))
            # 输入门控 maha（对退化后的输入）
            exp_cls = "brain_T1" if src_mod_t1(args.task) else "brain_T2"
            gv = gate.classify(x, exp_cls)
            per.append({
                "subject": r["subject"], "z": r["z"], "kind": kind, "dose": dose,
                "body_psnr": round(psnr, 2),
                "trust_mean": round(float(tr[mask].mean()), 4),
                "auroc": round(auroc_low_vs_higherr(tr[mask], err[mask]), 4),
                "decile_sep": round(float(
                    err[mask][tr[mask] <= np.percentile(tr[mask], 10)].mean() /
                    max(err[mask][tr[mask] >= np.percentile(tr[mask], 90)].mean(), 1e-9)), 2),
                "maha": gv.get("maha") if gv.get("maha") is not None else "",
                "gate_verdict": gv["verdict"],
            })
        if (i + 1) % 5 == 0 or i == len(rows) - 1:
            print(f"  [{i+1}/{len(rows)}] {(time.time()-t00)/60:.1f}min", flush=True)

    with open(os.path.join(run_dir, "per_case.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader(); w.writerows(per)

    # 剂量-响应汇总（每条件均值）
    summary = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "task": args.task,
               "n_slices": len(rows), "seed": args.seed, "conditions": {}}
    for kind, dose in CONDITIONS:
        rs = [q for q in per if q["kind"] == kind and q["dose"] == dose]
        summary["conditions"][f"{kind}_{dose}"] = {
            "body_psnr": round(float(np.mean([q["body_psnr"] for q in rs])), 2),
            "trust_mean": round(float(np.mean([q["trust_mean"] for q in rs])), 4),
            "auroc": round(float(np.nanmean([q["auroc"] for q in rs])), 4),
            "decile_sep": round(float(np.mean([q["decile_sep"] for q in rs])), 2),
            "maha_mean": round(float(np.mean([q["maha"] for q in rs if q["maha"] != ""])), 1)
            if any(q["maha"] != "" for q in rs) else None,
            "ood_rate": round(float(np.mean([q["gate_verdict"] == "OOD" for q in rs])), 3),
        }
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary["conditions"], ensure_ascii=False, indent=1))
    print("->", run_dir)


def src_mod_t1(task):
    return task.startswith("T1->")


if __name__ == "__main__":
    main()
