"""级联 OOD 探针：FeTS21 FLAIR → (官方 brats_flair_t2) → 合成T2 → (IXI自训 self_t2t1) → 信任图。

检验故事：健康脑上训练的模型，见到含肿瘤的 T2 输入时，信任图是否在肿瘤区下降。
"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trustbridge.engine import BridgeEngine            # noqa: E402
from trustbridge.pipeline import translate_single      # noqa: E402


def auroc(score, pos):
    s = score.ravel().astype(np.float64)
    p = pos.ravel()
    npos, nneg = int(p.sum()), s.size - int(p.sum())
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1)
    return float((ranks[p].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "data", "brats_fets_slices", "manifest.csv"), encoding="utf-8"))
            if float(r["tumor_frac"]) > 0.02]
    # 每受试者取肿瘤最大的一片，共 8 片跨 8 个受试者
    best = {}
    for r in rows:
        if r["subject"] not in best or float(r["tumor_frac"]) > float(best[r["subject"]]["tumor_frac"]):
            best[r["subject"]] = r
    sel = sorted(best.values(), key=lambda r: r["subject"])[:8]

    eng_syn = BridgeEngine("BRATS_FLAIR->T2")
    eng_ixi = BridgeEngine("T2->T1*", ckpt_path=os.path.join(ROOT, "checkpoints", "self_t2t1.ckpt"))
    print(f"探针 {len(sel)} 片: FLAIR→合成T2(官方) → T2→T1(IXI自训128px)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = []
    for r in sel:
        case = np.load(r["case_path"]).astype(np.float32)
        lab = np.load(r["label_path"])
        brain = case > 0.02
        tumor = (lab > 0) & brain

        syn_t2 = translate_single(eng_syn, case, n_ensemble=1, seed=7)["pred01"]
        out = translate_single(eng_ixi, syn_t2, n_ensemble=4, seed=7, gt01=None)
        res = out["result"]
        trust = res.trust_map
        lt = 1.0 - trust

        # 信任图是 128 模型域回到 240：translate_single 已对齐到输入形状
        a_fused = auroc(lt[brain], tumor[brain])
        a_rcrf = auroc(res.rcrf_map[brain], tumor[brain])
        a_unc = auroc(res.unc_map[brain], tumor[brain])
        t_in, t_out = float(trust[tumor].mean()), float(trust[brain & ~tumor].mean())
        results.append((r["subject"], r["z"], a_fused, a_rcrf, a_unc, t_in, t_out))
        print(f"{r['subject']} z{r['z']}: AUROC fused {a_fused:.3f} rcrf {a_rcrf:.3f} "
              f"unc {a_unc:.3f} | trust {t_in:.3f}in/{t_out:.3f}out")

        fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
        axes[0].imshow(case, cmap="gray"); axes[0].set_title("FLAIR")
        axes[1].imshow(syn_t2, cmap="gray"); axes[1].set_title("合成T2")
        axes[2].imshow(trust, cmap="RdYlGn", vmin=0, vmax=1); axes[2].set_title(f"IXI模型信任图 AUROC={a_fused:.2f}")
        axes[3].imshow(syn_t2, cmap="gray"); axes[3].contour(tumor, colors="red", linewidths=1.2)
        axes[3].set_title("肿瘤轮廓")
        for a in axes:
            a.axis("off")
        fig.tight_layout()
        os.makedirs(os.path.join(ROOT, "data", "_probe"), exist_ok=True)
        fig.savefig(os.path.join(ROOT, "data", "_probe", f"cascade_{r['subject'][-3:]}_z{r['z']}.png"), dpi=110)
        plt.close(fig)

    af = [x[2] for x in results]
    ar = [x[3] for x in results]
    au = [x[4] for x in results]
    print(f"\nmean AUROC: fused {np.mean(af):.3f} rcrf {np.mean(ar):.3f} unc {np.mean(au):.3f}")


if __name__ == "__main__":
    main()
