"""论文全部图表生成（英文版，输出 docs/paper_figures/）。

  fig1_overview.pdf/.png   方法总览示意（新绘，matplotlib 矢量）
  fig2_reverse_chain.png   反向采样链实拍（真实模型推理，英文标尺）
  fig3_trust_example.png   信任图实例六联（真实模型推理，英文标尺）
  fig4_reliability.png     可靠性曲线（校准分箱，双方向）
  fig5_ablation.png        消融柱状图（7 配置双方向）
  fig6_tumor.png           瘤区分析（信任/误差箱线 + 逐片 AUROC 分布）

用法: python scripts/make_paper_figures.py [1|2|3|4|5|6|all]
"""
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "docs", "paper_figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
})
BLUE, LBLUE, AMBER, GREEN, GRAY, RED = "#2563eb", "#93c5fd", "#f59e0b", "#16a34a", "#94a3b8", "#b91c1c"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def load_test_slice(idx=60):
    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_v3.csv"), encoding="utf-8")))
    r = [x for x in man if x["split"] == "test"][idx]
    return (np.load(r["t2_path"]).astype(np.float32),
            np.load(r["t1_path"]).astype(np.float32), r)


# ================= Fig.1 方法总览（纯示意，新绘） =================
def fig1():
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    ax.set_xlim(0, 124); ax.set_ylim(0, 56); ax.axis("off")

    def box(x, y, w, h, text, fc="#eff6ff", ec="#2563eb", fs=10.5, bold=False, ls="-"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                                   linewidth=1.4, linestyle=ls, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color="#0f172a", fontweight="bold" if bold else "normal", zorder=3)

    def arrow(x1, y1, x2, y2, color="#2563eb", style="-|>", lw=1.6, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls))

    # 主链
    box(2, 30, 14, 9, "Source image\n$y$ (T2-weighted)", fc="#f8fafc", ec=GRAY)
    box(21, 30, 20, 9, "Forward bridge\nconvex combination +\nsoft-prior noise", bold=False)
    box(46, 30, 16, 9, "$x_T = y_\\epsilon$\n(noisy source)")
    box(67, 26, 24, 17, "Reverse chain  $t=T\\!\\to\\!1$\n\nper step: self-consistent\nrecursion\n$\\tilde{x}_0^{(r+1)}=G_\\theta(x_t,t,y,\\tilde{x}_0^{(r)})$",
        fc="#eff6ff", ec="#1d4ed8", fs=10)
    box(96, 30, 14, 9, "Translated\n$\\hat{x}_0$ (T1-weighted)", fc="#f0fdf4", ec=GREEN)
    arrow(16.5, 34.5, 20.5, 34.5); arrow(41.5, 34.5, 45.5, 34.5)
    arrow(62.5, 34.5, 66.5, 34.5); arrow(91.5, 34.5, 95.5, 34.5)

    # 递归回环
    ax.annotate("", xy=(70, 43.5), xytext=(88, 43.5),
                arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=1.5,
                                connectionstyle="arc3,rad=-0.45", linestyle="--"))
    ax.text(79, 49.5, "repeat until self-consistent ($R$ recursions)",
            ha="center", fontsize=9, color="#7c3aed", style="italic")

    # 信号出口 1：递归残差
    box(58, 12, 22, 9, "Signal 1: per-pixel recursion\nresidual (last $K$ steps, avg. over chains)\n$R(p)=\\langle|\\tilde{x}_0^{(r+1)}(p)-\\tilde{x}_0^{(r)}(p)|\\rangle$",
        fc="#fef3c7", ec=AMBER, fs=9)
    arrow(79, 25.5, 72, 21.5, color=AMBER, ls="--")

    # 信号出口 2：集成分歧
    box(85, 12, 22, 9, "Signal 2: ensemble divergence\nper-pixel std over $N$ chains\n$U(p)=\\mathrm{std}_i\\,\\hat{x}_{0,i}(p)$",
        fc="#fef3c7", ec=AMBER, fs=9)
    arrow(103, 29.5, 100, 21.5, color=AMBER, ls="--")

    # 融合
    box(63, 1.5, 26, 8, "Trust map  $T(p)=1-w_1\\hat{R}(p)-w_2\\hat{U}(p)$\npercentile-normalized, fused", fc="#eff6ff", ec="#1d4ed8", bold=False)
    arrow(69, 11.5, 71, 10, color=AMBER, ls="--"); arrow(96, 11.5, 83, 10, color=AMBER, ls="--")
    box(94, 1.5, 20, 8, "Trust Score (0–100)\nQC gate: PASS /\nREVIEW / REJECT", fc="#f0fdf4", ec=GREEN, fs=9.5)
    arrow(89.5, 5.5, 93.5, 5.5)

    # 输入门控（虚线）
    box(2, 46, 20, 7, "Input modality gate\n(detects wrong-modality misuse)",
        fc="#fafafa", ec=GRAY, fs=9, ls="--")
    arrow(12, 45.5, 12, 39.8, color=GRAY, ls="--")

    ax.text(2, 54.5, "translation model (frozen, SelfRDB)", fontsize=9.5, color="#475569")
    ax.text(63, 54.5, "trust-quantification layer (training-free, annotation-free)",
            fontsize=9.5, color="#b45309")
    fig.tight_layout()
    save(fig, "fig1_overview")


# ================= Fig.2 反向链 + Fig.3 信任图实例（真实推理） =================
def fig23():
    import torch
    from trustbridge.engine import BridgeEngine
    from trustbridge.pipeline import translate_single

    eng = BridgeEngine("T2->T1")            # 官方 IXI 权重 256²
    t2, t1_gt, _ = load_test_slice(60)
    y = eng.preprocess(t2)
    d = eng.diffusion
    T = eng.n_steps
    dev = y.device

    with torch.inference_mode():
        ts_T = torch.full((1,), T, device=dev, dtype=torch.long)
        x_t = d.q_sample(ts_T, torch.zeros_like(y), y)
        cap = {8, 5, 3}
        rev, x0_r = [], None
        for t in range(T, 0, -1):
            tt = torch.full((1,), t, device=dev, dtype=torch.long)
            x0_r = torch.zeros_like(x_t)
            for _ in range(eng.max_recursions):
                x0_r = eng.generator(torch.cat((x_t, y), dim=1), tt, x_r=x0_r)
            if t in cap:
                rev.append((t, x0_r.clone()))
            x_t = d.q_posterior(tt, x_t, x0_r, y)
        final = x0_r

    to01 = lambda t_: eng.postprocess(t_)

    # ---- Fig.2 ----
    n = len(rev) + 3
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 2.6))
    axes[0].imshow(t2, cmap="gray"); axes[0].set_title("Input $y$ (T2)", fontsize=10)
    for ax, (t, x0e) in zip(axes[1:1 + len(rev)], rev):
        ax.imshow(to01(x0e), cmap="gray")
        ax.set_title(f"$t={t}$: target estimate", fontsize=9.5)
    axes[-2].imshow(to01(final), cmap="gray")
    axes[-2].set_title("Translation $\\hat{x}_0$ (T1)", fontsize=10, color=RED)
    axes[-1].imshow(t1_gt, cmap="gray"); axes[-1].set_title("Ground truth (T1)", fontsize=10, color=GREEN)
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    save(fig, "fig2_reverse_chain")

    # ---- Fig.3 ----
    res = translate_single(eng, t2, n_ensemble=4, step_skip=1, seed=0)["result"]
    pred01 = eng.postprocess(res.pred, t2.shape)
    if res.trust_map.shape != t2.shape:      # 256 画布中心裁剪回 128
        S = res.trust_map.shape[0]; H, W = t2.shape
        pt, pl = (S - H) // 2, (S - W) // 2
        crop = lambda m: np.ascontiguousarray(m[pt:pt + H, pl:pl + W])
        res.rcrf_map, res.unc_map, res.trust_map = crop(res.rcrf_map), crop(res.unc_map), crop(res.trust_map)
    err = np.abs(t1_gt - pred01)
    panels = [
        (t2, "Source $y$ (T2)", "gray", 0, 1),
        (pred01, "Translation $\\hat{x}_0$", "gray", 0, 1),
        (res.rcrf_map, "Recursion residual $R(p)$", "inferno", 0, 1),
        (res.unc_map, "Ensemble divergence $U(p)$", "magma", 0, 1),
        (res.trust_map, "Trust map $T(p)$", "inferno", 0, 1),
        (err, "Abs. error $|\\hat{x}_0-x_0|$ (ref.)", "hot", 0, max(err.max(), 1e-6)),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(2.15 * 6, 2.5))
    for ax, (img, ti, cmap, lo, hi) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi); ax.set_title(ti, fontsize=9.5); ax.axis("off")
    fig.tight_layout()
    save(fig, "fig3_trust_example")


# ================= Fig.4 可靠性曲线（校准分箱） =================
def fig4():
    d = json.load(open(os.path.join(ROOT, "results", "2026-09-27_calibration",
                                    "summary.json"), encoding="utf-8"))["directions"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=False)
    for ax, (key, title) in zip(axes, [("T2->T1", "T2$\\to$T1"), ("T1->T2", "T1$\\to$T2")]):
        dd = d[key]
        x = np.arange(1, 11)
        ax.bar(x, dd["bin_abs_err"], color=BLUE, alpha=.85, width=.72)
        for xi, v in zip(x, dd["bin_abs_err"]):
            ax.text(xi, v * 1.06 + 0.008, f"{v:.2f}", ha="center", fontsize=8.5, color="#1e3a8a")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:.2f}" for t in dd["bin_trust"]], rotation=45, fontsize=8)
        ax.set_xlabel("mean trust value in decile bin (low $\\to$ high)")
        ax.set_ylabel("mean abs. error in bin")
        ax.set_title(f"{title}   (monotonicity {dd['monotonicity']:.2f}, "
                     f"separation {dd['separation_ratio']:.0f}$\\times$)", fontsize=11, pad=14)
        ax.set_ylim(0, max(dd["bin_abs_err"]) * 1.22)
        ax.grid(axis="y", alpha=.3)
    fig.suptitle("Reliability: mean error per trust decile (80 slices per direction)", fontsize=12)
    fig.tight_layout()
    save(fig, "fig4_reliability")


# ================= Fig.5 消融柱状图（7 配置双方向） =================
def fig5():
    keys = [("gain_random_db", "Random", GRAY),
            ("gain_gradient_db", "Gradient", "#94a3b8"),
            ("gain_input_diff_db", "Input-diff$^{\\dagger}$", "#cbd5e1"),
            ("gain_rcrf_only_db", "RCRF only", "#60a5fa"),
            ("gain_unc_only_db", "Ensemble only", "#3b82f6"),
            ("gain_fused_db", "Fusion (ours)", BLUE),
            ("gain_oracle_db", "Oracle$^{\\ddagger}$", GREEN)]
    sums = {}
    for task, tag in [("T2->T1", "T22T1"), ("T1->T2", "T12T2")]:
        p = os.path.join(ROOT, "results", f"2026-09-23_{tag}_self_test", "summary.json")
        sums[task] = json.load(open(p, encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.9), sharey=False)
    for ax, (task, title) in zip(axes, [("T2->T1", "T2$\\to$T1 (IXI)"), ("T1->T2", "T1$\\to$T2 (IXI)")]):
        vals = [sums[task][k]["mean"] for k, _, _ in keys]
        colors = [c for _, _, c in keys]
        x = np.arange(len(keys))
        bars = ax.bar(x, vals, color=colors, width=.68,
                      edgecolor=["#64748b" if "Input" in n else "none" for _, n, _ in keys])
        bars[2].set_hatch("//")
        bars[-1].set_hatch("xx")
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.18, f"{v:+.2f}", ha="center", fontsize=9,
                    color="#0f172a", fontweight="bold" if "Fusion" in keys[xi][1] else "normal")
        ax.axhline(0, color="#475569", lw=.8)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n, _ in keys], rotation=24, ha="right", fontsize=9)
        ax.set_ylabel("$\\Delta$PSNR (dB) after dropping\nleast-trusted 20% pixels")
        ax.set_title(title, fontsize=11.5)
        ax.grid(axis="y", alpha=.3)
        lo = min(min(vals) - 0.5, -0.6); hi = max(vals) + 1.2
        ax.set_ylim(lo, hi)
    fig.suptitle("Ablation: sparsification gain by trust-signal configuration (60 test slices each)", fontsize=12)
    fig.text(0.99, 0.01, "$^{\\dagger}$GT-leaking reference   $^{\\ddagger}$oracle upper bound (uses GT)",
             ha="right", fontsize=9, color="#64748b")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save(fig, "fig5_ablation")


# ================= Fig.6 瘤区分析（箱线 + 逐片 AUROC 分布） =================
def fig6():
    """病灶分析（修正版）：体掩码内 AUROC≈随机（信任盲区）+ 误差集中 + 自信幻觉案例。"""
    import csv as _csv
    run = os.path.join(ROOT, "results", "2026-11-12_tumor_trust_corrected")
    rows = list(_csv.DictReader(open(os.path.join(run, "per_slice.csv"), encoding="utf-8")))
    t_in = np.array([float(r["t_in"]) for r in rows])
    t_out = np.array([float(r["t_out"]) for r in rows])
    e_in = np.array([float(r["e_in"]) for r in rows])
    e_out = np.array([float(r["e_out"]) for r in rows])
    au = np.array([float(r["auroc_lowtrust"]) for r in rows])

    st = np.load(os.path.join(ROOT, "outputs", "_bt0.npy"))
    flair, trust_c, seg, pred_c = st[0], st[1], st[2].astype(bool), st[3]

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4))
    (a, b, c), (d, e, f) = axes

    bp = a.boxplot([t_in, t_out], tick_labels=["inside tumor", "body minus tumor"],
                   widths=.5, patch_artist=True, medianprops=dict(color="#0f172a"))
    for patch, col in zip(bp["boxes"], [AMBER, BLUE]):
        patch.set_facecolor(col); patch.set_alpha(.55)
    a.text(1.28, t_in.mean(), f"{t_in.mean():.3f}", fontsize=9.5, va="center")
    a.text(2.28, t_out.mean(), f"{t_out.mean():.3f}", fontsize=9.5, va="center")
    a.set_ylabel("mean trust per slice"); a.set_title("(a) Trust: inside vs outside tumor", fontsize=11)
    a.grid(axis="y", alpha=.3)

    bp = b.boxplot([e_in, e_out], tick_labels=["inside tumor", "body minus tumor"],
                   widths=.5, patch_artist=True, medianprops=dict(color="#0f172a"))
    for patch, col in zip(bp["boxes"], [AMBER, BLUE]):
        patch.set_facecolor(col); patch.set_alpha(.55)
    b.text(1.28, e_in.mean(), f"{e_in.mean():.4f}", fontsize=9.5, va="center")
    b.text(2.28, e_out.mean(), f"{e_out.mean():.4f}", fontsize=9.5, va="center")
    b.set_ylabel("mean abs. error per slice")
    b.set_title(f"(b) Error: {e_in.mean()/max(e_out.mean(),1e-9):.2f}$\times$ inside (90% of slices)", fontsize=11)
    b.grid(axis="y", alpha=.3)

    c.hist(au, bins=20, color=BLUE, alpha=.8)
    c.axvline(au.mean(), color=RED, lw=1.8)
    c.text(au.mean() + .015, c.get_ylim()[1] * .88, f"mean {au.mean():.2f}", color=RED, fontsize=10)
    c.axvline(0.5, color=GRAY, ls="--", lw=1.2)
    c.text(0.508, c.get_ylim()[1] * .55, "chance", color=GRAY, fontsize=9, rotation=90)
    c.set_xlabel("within-body AUROC (low trust $\to$ tumor)")
    c.set_ylabel("# slices"); c.set_title("(c) Trust map carries NO tumor information", fontsize=11)
    c.grid(axis="y", alpha=.3)

    d.imshow(flair, cmap="gray"); d.set_title("FLAIR source (tumor bright, right)", fontsize=10.5)
    e.imshow(trust_c, cmap="inferno", vmin=0, vmax=1)
    e.contour(seg, levels=[0.5], colors="#22d3ee", linewidths=1.8)
    e.set_title("Trust map + tumor contour:\ntumor is HIGH-trust (confidently wrong)", fontsize=10)
    f.imshow(pred_c, cmap="gray"); f.set_title("Translated T2 (smooth fill of tumor)", fontsize=10.5)
    for ax in (d, e, f): ax.axis("off")

    fig.suptitle("Lesion analysis on BraTS2018 (corrected, within-body AUROC): the trust map is blind to the tumor "
                 "--- a confidently-wrong hallucination", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig6_tumor")




if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"):
        fig1()
    if which in ("all", "23"):
        fig23()
    if which in ("all", "4"):
        fig4()
    if which in ("all", "5"):
        fig5()
    if which in ("all", "6"):
        fig6()
    print("ALL PAPER FIGURES DONE")
