# -*- coding: utf-8 -*-
"""论文 v0.6 图表生成（无 GPU 部分）：
fig1_overview_v2  方法总览（无门控/TrustScore/QC 框，蓝橙配色）
fig3_collapse     崩塌 vs 健康模型信任分布（collapse_alarm 数据）
fig4_body_decile  体区十分箱可靠性（body_decile 数据，脚本跑完后重跑本脚本）
输出: docs/latex/figures/*.pdf（矢量，300dpi 等效）
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "latex", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
C_BLUE, C_ORANGE, C_GRAY = "#2563eb", "#f59e0b", "#64748b"


# ---------------- fig1 方法总览 ----------------
def fig_overview():
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eff6ff", ec=C_BLUE, fs=10, bold=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                                   linewidth=1.6, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, style="-", color=C_BLUE):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6,
                                    linestyle=style))

    # 主链（y=58..78）
    box(0.5, 58, 16, 20, "Source image $y$\n(T2-weighted)", bold=True, fs=9.5)
    box(19.5, 58, 18, 20, "Frozen SelfRDB bridge\nconvex combination +\nsoft-prior noise", fs=9.5)
    box(40.5, 58, 14, 20, "$x_T = y_\\epsilon$\n(noisy source)", fs=9.5)
    box(57.5, 55, 26, 26, "Reverse chain $t=T\\to 1$\nper-step self-consistent recursion\n$\\tilde{x}_0^{(r+1)}=G_\\theta(x_t,t,y,\\tilde{x}_0^{(r)})$",
        fc="#dbeafe", bold=True, fs=9.5)
    box(86.5, 58, 13, 20, "Translated\n$\\hat{x}_0$ (T1)", fc="#dcfce7", ec="#15803d", bold=True, fs=9.5)
    arrow(16.5, 68, 19.5, 68); arrow(37.5, 68, 40.5, 68); arrow(54.5, 68, 57.5, 68)
    arrow(83.5, 68, 86.5, 68)
    # 递归自环
    ax.annotate("", xy=(80, 82), xytext=(63, 82),
                arrowprops=dict(arrowstyle="-|>", color="#9333ea", lw=1.4,
                                connectionstyle="arc3,rad=-0.4", linestyle="--"))
    ax.text(71.5, 90, "repeat until self-consistent ($R$ recursions)",
            ha="center", fontsize=9, color="#9333ea", style="italic")

    # 信号 1/2（y=24..42）
    box(48, 22, 24, 20, "Signal 1 · convergence residual field\nCReF$(p)$: per-pixel gap between\nsuccessive recursive estimates,\nlast $K$ steps, averaged over chains",
        fc="#fef3c7", ec=C_ORANGE, fs=8.4)
    box(75, 22, 24, 20, "Signal 2 · ensemble divergence\n$U(p)$: per-pixel std across $N$\nindependent bridge chains",
        fc="#fef3c7", ec=C_ORANGE, fs=8.4)
    arrow(64, 55, 60, 42, style="--", color=C_ORANGE)
    arrow(83, 55, 87, 42, style="--", color=C_ORANGE)

    # 融合（y=0..13）
    box(48, 0, 51, 13, "Trust map  $T(p) = 1 - w_1\\,\\widehat{\\mathrm{CReF}}(p) - w_2\\,\\widehat{U}(p)$   (percentile-normalized, $w_1{=}0.6$ / $w_2{=}0.4$)",
        fc="#eff6ff", bold=True, fs=9.5)
    arrow(60, 22, 63, 13, style="--", color=C_ORANGE)
    arrow(87, 22, 84, 13, style="--", color=C_ORANGE)

    ax.text(50, 99, "translation model (frozen, SelfRDB)", ha="center",
            fontsize=10, color=C_GRAY)
    ax.text(50, -16, "The two signals are already computed by the sampler; harvesting them adds no model, no training and no labels.\n"
                    "Endpoint noise and generator latents provide the ensemble diversity.",
            ha="center", fontsize=8.6, color=C_GRAY)
    ax.set_xlim(0, 101); ax.set_ylim(-24, 101)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_overview_v2.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig1_overview_v2.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig1 done")


# ---------------- fig3 崩塌 vs 健康 ----------------
def fig_collapse():
    rows = list(csv.DictReader(open(os.path.join(
        ROOT, "results", "2026-11-04_collapse_alarm", "per_case.csv"), encoding="utf-8")))
    official = np.array([float(r["trust_score"]) for r in rows if r["model"] == "official_15db"])
    self_m = np.array([float(r["trust_score"]) for r in rows if r["model"] == "self_20db"])

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    bins = np.linspace(55, 100, 24)
    ax.hist(official, bins=bins, alpha=0.75, color=C_ORANGE,
            label=f"official weights (collapsed, 12.98 dB)\nmean trust {official.mean():.1f}, all PASS",
            edgecolor="white")
    ax.hist(self_m, bins=bins, alpha=0.75, color=C_BLUE,
            label=f"self-trained (healthy, 18.24 dB)\nmean trust {self_m.mean():.1f}",
            edgecolor="white")
    ax.axvline(official.mean(), color=C_ORANGE, lw=1.4, linestyle="--")
    ax.axvline(self_m.mean(), color=C_BLUE, lw=1.4, linestyle="--")
    ax.set_xlabel("image-level Trust Score (body mean, 0-100)")
    ax.set_ylabel("# test slices")
    ax.set_title("A collapsed base model is MORE confident than a healthy one:\n"
                 "variance-type trust signals cannot see the failure (20 test slices each)")
    ax.legend(fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_collapse_vs_healthy.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_collapse_vs_healthy.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig_collapse done")


# ---------------- fig4 体区十分箱 ----------------
def fig_body_decile():
    p = os.path.join(ROOT, "results", "2026-11-08_body_decile", "per_case.csv")
    if not os.path.exists(p):
        p = os.path.join(ROOT, "results", "2026-11-05_body_decile", "per_case.csv")
    if not os.path.exists(p):
        p = os.path.join(ROOT, "results", "2026-11-04_body_decile", "per_case.csv")
    if not os.path.exists(p):
        print("body_decile data not ready yet - rerun this script later"); return
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=False)
    for ax, task, title in zip(axes, ("T1->T2*", "T2->T1*"),
                               ("T1→T2 (self-trained)", "T2→T1 (self-trained)")):
        rs = [q for q in rows if q["task"] == task]
        med = [float(np.median([float(q[f"bin{b}"]) for q in rs])) for b in range(1, 11)]
        q25 = [float(np.percentile([float(q[f"bin{b}"]) for q in rs], 25)) for b in range(1, 11)]
        q75 = [float(np.percentile([float(q[f"bin{b}"]) for q in rs], 75)) for b in range(1, 11)]
        x = np.arange(1, 11)
        ax.bar(x, med, color=C_BLUE, alpha=0.85)
        ax.errorbar(x, med, yerr=[np.array(med) - np.array(q25), np.array(q75) - np.array(med)],
                    fmt="none", ecolor="#1e3a8a", elinewidth=1, capsize=3)
        sep = med[0] / max(med[-1], 1e-9)
        ax.set_title(f"{title}\nwithin-body decile separation {sep:.1f}×")
        ax.set_xlabel("trust decile bin (low → high)")
        ax.set_xticks(x)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean abs. error in bin\n(within body)")
    fig.suptitle("Within-body reliability: mean absolute error per trust decile\n"
                 "(60 test slices per direction, seed 2026; error bars = IQR)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "fig_body_decile.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_body_decile.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("fig_body_decile done")


if __name__ == "__main__":
    fig_overview()
    fig_collapse()
    fig_body_decile()
