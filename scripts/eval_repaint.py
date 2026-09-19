"""RePaint 式区域重采样 vs uniform 基线：同预算对照实验（GPU 空闲后运行）。

对照设计（等效预算对齐，均为 7 次生成器链调用）：
  uniform   : 4 条完整主链取均值（官方语义）
  repaint   : 3 条主链（打底均值）+ 中点 t* 后 4 条重试子链（仅不确定块采用重试均值）

指标：
  整体 PSNR/SSIM（官方口径）
  不确定块内 PSNR（RePaint 收益应集中于此）
  不确定块定位 AUROC（块级分歧 vs 块级误差，纯基线能力）

用法（训练结束、GPU 空闲后）:
  python scripts/eval_repaint.py --task T1->T2 --ckpt checkpoints/self_t1t2.ckpt --n 30
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # noqa: E402

from trustbridge.engine import BridgeEngine, TASKS  # noqa: E402
from trustbridge.metrics import mean_norm  # noqa: E402
from trustbridge.repaint import RepaintSampler  # noqa: E402
from trustbridge.sampler import TrustSampler  # noqa: E402


def psnr_ssim(gt, pr):
    g, p = mean_norm(gt), mean_norm(pr)
    return (peak_signal_noise_ratio(g, p, data_range=g.max()),
            structural_similarity(g, p, data_range=g.max()) * 100)


def block_map(x, block):
    H, W = x.shape
    nb = H // block
    return x[: nb * block, : nb * block].reshape(nb, block, nb, block).mean(axis=(1, 3))


def block_mask_up(low_blk, block, H, W):
    up = low_blk.repeat_interleave(block, dim=2).repeat_interleave(block, dim=3)
    return up[:, :H, :W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T1->T2")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_v3.csv"), encoding="utf-8")))
    rows = [r for r in man if r["split"] == "test"]
    step = max(1, len(rows) // args.n)
    rows = rows[::step][: args.n]

    eng = BridgeEngine(args.task, ckpt_path=args.ckpt)
    _, src_mod, tgt_mod, _ = TASKS[args.task]
    rp = RepaintSampler(eng)
    uni = TrustSampler(eng)

    acc = {"uniform": {"psnr": [], "ssim": [], "inblk": []},
           "repaint": {"psnr": [], "ssim": [], "inblk": []}}
    aurocs = []

    for i, r in enumerate(rows):
        src = np.load(r["t2_path"] if src_mod == "T2" else r["t1_path"]).astype(np.float32)
        gt = np.load(r["t1_path"] if tgt_mod == "T1" else r["t2_path"]).astype(np.float32)
        y = eng.preprocess(src)

        # uniform 基线（4 主链）
        u = uni.sample(y, n_ensemble=4, step_skip=1, seed=args.seed)
        u_img = eng.postprocess(u.pred, src.shape)
        # repaint（3 主链 + 4 重试子链；不确定块=主链分歧 top 25%）
        rr = rp.sample_repaint(y, n_main=3, j_retry=4, step_skip=1, seed=args.seed)
        r_img = eng.postprocess(rr.pred, src.shape)

        # 不确定块掩码（来自 repaint 的低信任块，两方案统一用块级误差评估）
        low_blk = torch.tensor(rr.low_px)[None]
        mup = block_mask_up(low_blk, 8, *src.shape).numpy()
        blk_ok = mup.sum() > 200

        for name, img in (("uniform", u_img), ("repaint", r_img)):
            p, s = psnr_ssim(gt, img)
            acc[name]["psnr"].append(p); acc[name]["ssim"].append(s)
            if blk_ok:
                g, pr = mean_norm(gt), mean_norm(img)
                m = mup.astype(bool)
                acc[name]["inblk"].append(
                    peak_signal_noise_ratio(g[m], pr[m], data_range=g.max()))

        # 定位 AUROC（块级分歧 vs 块级误差）
        blk_err = block_map(np.abs(gt - u_img), 8)
        blk_unc = block_map(np.abs(eng.postprocess(u.members[0], src.shape) -
                                   eng.postprocess(u.members[1], src.shape)), 8)
        if blk_err.std() > 1e-6 and blk_unc.std() > 1e-6:
            rho = spearmanr(blk_unc.ravel(), blk_err.ravel()).statistic
            aurocs.append(float(rho))

        if (i + 1) % 5 == 0 or i == len(rows) - 1:
            print(f"  [{i+1}/{len(rows)}] uniform {np.mean(acc['uniform']['psnr']):.2f} | "
                  f"repaint {np.mean(acc['repaint']['psnr']):.2f}", flush=True)

    def pack(name):
        return {"psnr": round(float(np.mean(acc[name]["psnr"])), 2),
                "ssim": round(float(np.mean(acc[name]["ssim"])), 2),
                "inblk_psnr": (round(float(np.mean(acc[name]["inblk"])), 2)
                               if acc[name]["inblk"] else None)}

    res = {"task": args.task, "n_slices": len(rows), "seed": args.ckpt,
           "uniform": pack("uniform"), "repaint": pack("repaint"),
           "repaint_gain_psnr": round(res_p := (float(np.mean(acc["repaint"]["psnr"])) -
                                                float(np.mean(acc["uniform"]["psnr"]))), 3),
           "blk_spearman_mean": round(float(np.mean(aurocs)), 4) if aurocs else None}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    out = args.out or os.path.join(ROOT, "outputs", f"repaint_{args.task.replace('->','_')}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("->", out)


if __name__ == "__main__":
    main()
