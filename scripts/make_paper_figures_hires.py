"""论文图高清版：fig2/fig3 用 256² 原生切片重推理（600dpi），fig1 用 SVG 矢量重绘。

输出 docs/paper_figures/：
  fig1_overview  手绘 SVG（学术蓝）→ PDF + 600dpi PNG（附可编辑 SVG 源，Inkscape/Illustrator 可改）
  fig2_reverse_chain / fig3_trust_example  256² 原生切片、官方权重、dpi=600

用法: python scripts/make_paper_figures_hires.py [1|23|all]
"""
import csv
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

import torch  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.unicode_minus": False})
DPI = 600
BLUE, RED, GREEN, AMBER = "#2563eb", "#b91c1c", "#16a34a", "#f59e0b"

# tspan 便捷构造：下标/上标（SVG 1.1 稳定做法，避免 Unicode 下标字母缺字）
def sub(t):   return f'<tspan dy="4" font-size="10">{t}</tspan>'
def sup(t):   return f'<tspan dy="-11" font-size="10">{t}</tspan>'
def back(t):  return f'<tspan dy="7" >{t}</tspan>'      # 从上标回基线
def back2(t): return f'<tspan dy="-4" >{t}</tspan>'     # 从下标回基线


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=DPI, bbox_inches="tight",
                facecolor="white")
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name, "(png@600dpi + pdf)")


def load_256_test(idx=60):
    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "ixi_slices",
                                                "manifest_256.csv"), encoding="utf-8")))
    rows = [r for r in man if r["split"] == "test"]
    r = rows[idx]
    return (np.load(r["t2_path"]).astype(np.float32),
            np.load(r["t1_path"]).astype(np.float32), r)


def fig23_hires():
    from trustbridge.engine import BridgeEngine
    from trustbridge.pipeline import translate_single

    eng = BridgeEngine("T2->T1")            # 官方权重，原生 256² 不再补零
    t2, t1_gt, r = load_256_test(60)
    print("切片:", r["subject"], "z=", r["z"], "shape=", t2.shape)
    y = eng.preprocess(t2)
    d = eng.diffusion
    T = eng.n_steps
    dev = y.device

    with torch.inference_mode():
        ts_T = torch.full((1,), T, device=dev, dtype=torch.long)
        x_t = d.q_sample(ts_T, torch.zeros_like(y), y)
        cap = {8, 5, 3}
        rev = []
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

    n = len(rev) + 3
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.7))
    axes[0].imshow(t2, cmap="gray"); axes[0].set_title("Input $y$ (T2)", fontsize=11)
    for ax, (t, x0e) in zip(axes[1:1 + len(rev)], rev):
        ax.imshow(to01(x0e), cmap="gray")
        ax.set_title(f"$t={t}$: target estimate", fontsize=10.5)
    axes[-2].imshow(to01(final), cmap="gray")
    axes[-2].set_title("Translation $\\hat{x}_0$ (T1)", fontsize=11, color=RED)
    axes[-1].imshow(t1_gt, cmap="gray"); axes[-1].set_title("Ground truth (T1)", fontsize=11, color=GREEN)
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    save(fig, "fig2_reverse_chain")

    res = translate_single(eng, t2, n_ensemble=4, step_skip=1, seed=0)["result"]
    pred01 = eng.postprocess(res.pred, t2.shape)
    err = np.abs(t1_gt - pred01)
    panels = [
        (t2, "Source $y$ (T2)", "gray", 0, 1),
        (pred01, "Translation $\\hat{x}_0$", "gray", 0, 1),
        (res.rcrf_map, "Recursion residual $R(p)$", "inferno", 0, 1),
        (res.unc_map, "Ensemble divergence $U(p)$", "magma", 0, 1),
        (res.trust_map, "Trust map $T(p)$", "inferno", 0, 1),
        (err, "Abs. error $|\\hat{x}_0-x_0|$ (ref.)", "hot", 0, max(err.max(), 1e-6)),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(2.3 * 6, 2.7))
    for ax, (img, ti, cmap, lo, hi) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi); ax.set_title(ti, fontsize=10.5); ax.axis("off")
    fig.tight_layout()
    save(fig, "fig3_trust_example")


# ================= Fig.1 SVG 矢量重绘 =================
SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="1240" height="560" viewBox="0 0 1240 560"
     font-family="Segoe UI, DejaVu Sans, Arial" font-size="15">
  <defs>
    <marker id="ab" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
      <path d="M0,0 L7,3.2 L0,6.4 Z" fill="#2563eb"/></marker>
    <marker id="aa" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
      <path d="M0,0 L7,3.2 L0,6.4 Z" fill="#b45309"/></marker>
    <marker id="ag" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
      <path d="M0,0 L7,3.2 L0,6.4 Z" fill="#94a3b8"/></marker>
    <marker id="ap" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">
      <path d="M0,0 L7,3.2 L0,6.4 Z" fill="#7c3aed"/></marker>
  </defs>
  <rect width="1240" height="560" fill="white"/>

  <text x="24" y="30" font-size="14" fill="#475569">translation model (frozen, SelfRDB)</text>
  <text x="640" y="30" font-size="14" fill="#b45309">trust-quantification layer (training-free, annotation-free)</text>

  <!-- 输入门控 -->
  <rect x="24" y="44" width="216" height="46" rx="10" fill="#fafafa" stroke="#94a3b8"
        stroke-width="1.4" stroke-dasharray="7 5"/>
  <text x="132" y="63" text-anchor="middle" fill="#334155" font-size="13.5">Input modality gate</text>
  <text x="132" y="81" text-anchor="middle" fill="#64748b" font-size="12">detects wrong-modality misuse</text>

  <!-- 主链 -->
  <rect x="24" y="170" width="150" height="62" rx="10" fill="#f8fafc" stroke="#64748b" stroke-width="1.5"/>
  <text x="99" y="196" text-anchor="middle" fill="#0f172a" font-weight="bold">Source image</text>
  <text x="99" y="216" text-anchor="middle" fill="#0f172a" font-weight="bold">y (T2-weighted)</text>

  <rect x="220" y="170" width="196" height="62" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>
  <text x="318" y="192" text-anchor="middle" fill="#0f172a" font-weight="bold">Forward bridge</text>
  <text x="318" y="210" text-anchor="middle" fill="#0f172a" font-size="13">convex combination +</text>
  <text x="318" y="227" text-anchor="middle" fill="#0f172a" font-size="13">soft-prior noise</text>

  <rect x="462" y="170" width="140" height="62" rx="10" fill="#eff6ff" stroke="#2563eb" stroke-width="1.5"/>
  <text x="532" y="192" text-anchor="middle" fill="#0f172a">x<tspan dy="4" font-size="10">T</tspan><tspan dy="-4"> = y</tspan><tspan dy="4" font-size="10">&#949;</tspan></text>
  <text x="532" y="216" text-anchor="middle" fill="#0f172a">(noisy source)</text>

  <rect x="648" y="152" width="270" height="98" rx="10" fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.8"/>
  <text x="783" y="176" text-anchor="middle" fill="#0f172a" font-weight="bold">Reverse chain  t = T &#8594; 1</text>
  <text x="783" y="198" text-anchor="middle" fill="#0f172a">per step: self-consistent recursion</text>
  <text x="783" y="228" text-anchor="middle" fill="#1d4ed8" font-style="italic">x&#771;<tspan dy="4" font-size="10">0</tspan><tspan dy="-14" font-size="10">(r+1)</tspan><tspan dy="10"> = G</tspan><tspan dy="4" font-size="10">&#952;</tspan><tspan dy="-4">(x</tspan><tspan dy="4" font-size="10">t</tspan><tspan dy="-4">, t, y, x&#771;</tspan><tspan dy="4" font-size="10">0</tspan><tspan dy="-14" font-size="10">(r)</tspan><tspan dy="10">)</tspan></text>

  <rect x="964" y="170" width="150" height="62" rx="10" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.5"/>
  <text x="1039" y="196" text-anchor="middle" fill="#0f172a" font-weight="bold">Translated image</text>
  <text x="1039" y="216" text-anchor="middle" fill="#0f172a" font-weight="bold">x&#770;<tspan dy="4" font-size="10">0</tspan><tspan dy="-4"> (T1-weighted)</tspan></text>

  <line x1="174" y1="201" x2="216" y2="201" stroke="#2563eb" stroke-width="2" marker-end="url(#ab)"/>
  <line x1="416" y1="201" x2="458" y2="201" stroke="#2563eb" stroke-width="2" marker-end="url(#ab)"/>
  <line x1="602" y1="201" x2="644" y2="201" stroke="#2563eb" stroke-width="2" marker-end="url(#ab)"/>
  <line x1="918" y1="201" x2="960" y2="201" stroke="#2563eb" stroke-width="2" marker-end="url(#ab)"/>

  <path d="M 740 148 Q 783 116 826 148" fill="none" stroke="#7c3aed" stroke-width="1.8"
        stroke-dasharray="6 5" marker-end="url(#ap)"/>
  <text x="783" y="108" text-anchor="middle" fill="#7c3aed" font-size="13" font-style="italic">
    repeat until self-consistent (R recursions)</text>

  <!-- 门控→主链 虚线 -->
  <line x1="132" y1="90" x2="99" y2="166" stroke="#94a3b8" stroke-width="1.4"
        stroke-dasharray="6 5" marker-end="url(#ag)"/>

  <!-- 信号出口 -->
  <rect x="600" y="300" width="300" height="92" rx="10" fill="#fef3c7" stroke="#b45309" stroke-width="1.6"/>
  <text x="750" y="326" text-anchor="middle" fill="#92400e" font-weight="bold" font-size="14">
    Signal 1 &#183; recursion residual field</text>
  <text x="750" y="350" text-anchor="middle" fill="#334155" font-size="13">
    per-pixel gap between successive estimates,</text>
  <text x="750" y="370" text-anchor="middle" fill="#334155" font-size="13">
    averaged over the last K steps and N chains</text>

  <rect x="930" y="300" width="280" height="92" rx="10" fill="#fef3c7" stroke="#b45309" stroke-width="1.6"/>
  <text x="1070" y="326" text-anchor="middle" fill="#92400e" font-weight="bold" font-size="14">
    Signal 2 &#183; ensemble divergence</text>
  <text x="1070" y="350" text-anchor="middle" fill="#334155" font-size="13">
    per-pixel std across N independent</text>
  <text x="1070" y="370" text-anchor="middle" fill="#334155" font-size="13">bridge chains</text>

  <line x1="760" y1="250" x2="736" y2="296" stroke="#b45309" stroke-width="1.8"
        stroke-dasharray="6 5" marker-end="url(#aa)"/>
  <line x1="1010" y1="232" x2="1058" y2="296" stroke="#b45309" stroke-width="1.8"
        stroke-dasharray="6 5" marker-end="url(#aa)"/>

  <!-- 融合 -->
  <rect x="700" y="440" width="290" height="60" rx="10" fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.8"/>
  <text x="845" y="465" text-anchor="middle" fill="#0f172a" font-weight="bold">
    Trust map  T(p) = 1 &#8722; w<tspan dy="4" font-size="10">1</tspan><tspan dy="-4">R&#770;(p) &#8722; w</tspan><tspan dy="4" font-size="10">2</tspan><tspan dy="-4">U&#770;(p)</tspan></text>
  <text x="845" y="487" text-anchor="middle" fill="#334155" font-size="13">
    percentile-normalized, w<tspan dy="3" font-size="10">1</tspan><tspan dy="-3">=0.6 / w</tspan><tspan dy="3" font-size="10">2</tspan><tspan dy="-3">=0.4</tspan></text>

  <line x1="760" y1="392" x2="790" y2="436" stroke="#b45309" stroke-width="1.8"
        stroke-dasharray="6 5" marker-end="url(#aa)"/>
  <line x1="1060" y1="392" x2="910" y2="436" stroke="#b45309" stroke-width="1.8"
        stroke-dasharray="6 5" marker-end="url(#aa)"/>

  <rect x="1030" y="440" width="180" height="60" rx="10" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.6"/>
  <text x="1120" y="465" text-anchor="middle" fill="#0f172a" font-weight="bold">Trust Score (0&#8211;100)</text>
  <text x="1120" y="487" text-anchor="middle" fill="#334155" font-size="13">QC: PASS / REVIEW / REJECT</text>

  <line x1="990" y1="470" x2="1026" y2="470" stroke="#2563eb" stroke-width="2" marker-end="url(#ab)"/>

  <text x="24" y="545" fill="#64748b" font-size="12.5">
    The two signals are already computed by the sampler; harvesting them adds no model,
    no training and no labels. Endpoint noise and generator latents provide the ensemble diversity.</text>
</svg>
"""


def fig1_svg():
    svg_path = os.path.join(OUT, "fig1_overview.svg")
    open(svg_path, "w", encoding="utf-8").write(SVG)
    from playwright.sync_api import sync_playwright
    html = ("<!DOCTYPE html><html><head><style>@page{size:1240px 560px;margin:0;}"
            "body{margin:0;}</style></head><body>" + SVG + "</body></html>")
    hp = os.path.join(OUT, "_fig1.html")
    open(hp, "w", encoding="utf-8").write(html)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge")
        page = b.new_page(viewport={"width": 1240, "height": 560},
                          device_scale_factor=5)
        page.goto("file:///" + hp.replace(chr(92), "/"))
        page.wait_for_timeout(800)
        page.screenshot(path=os.path.join(OUT, "fig1_overview.png"))
        page.pdf(path=os.path.join(OUT, "fig1_overview.pdf"),
                 width="1240px", height="560px", print_background=True,
                 margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        b.close()
    os.remove(hp)
    print("saved fig1_overview (svg source + png@600dpi + pdf)；SVG 可在 Inkscape/Illustrator/Figma 继续编辑")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "1"):
        fig1_svg()
    if which in ("all", "23"):
        fig23_hires()
    print("HIRES FIGURES DONE")
