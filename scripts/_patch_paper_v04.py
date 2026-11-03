# -*- coding: utf-8 -*-
"""论文 v0.4 结构性改写（不依赖新实验数字的部分）。运行一次即完成。"""
import io

fp = r"D:\MedTranslateProject\docs\latex\main.tex"
s = io.open(fp, encoding="utf-8").read()

# 1) 标题：采纳 Rethinking 框架
old = (
    "% 标题双轨：参赛/工作台用方法标题（下）；投刊（BSPC/CBM 档）建议换用框架标题：\n"
    '% "Rethinking Uncertainty in Medical Image Translation: Convergence Residuals,\n'
    "%  Evaluation Protocols, and Quality Control for Diffusion-Bridge Synthesis\"\n"
    "\\title{TrustBridge: Trust-Map Quantification for Diffusion-Bridge\\\\\n"
    "Medical Image Translation via Convergence Residual Fields}"
)
new = (
    "% 标题：采纳外部评审建议的 Rethinking 框架（揭示陷阱与划定边界）\n"
    "\\title{Rethinking Uncertainty Quantification in Diffusion-Bridge Medical Image\n"
    "Translation: Pitfalls, Baselines, and Boundaries\\\\\n"
    "\\large (system: TrustBridge --- a training-free trust layer with\n"
    "Convergence Residual Fields)}"
)
assert old in s, "title block not found"
s = s.replace(old, new)

# 2) 摘要重写：痛点先行
i0 = s.index("\\begin{abstract}")
i1 = s.index("\\end{abstract}") + len("\\end{abstract}")
new_abs = r"""\begin{abstract}
Two pitfalls quietly corrupt most uncertainty-quantification (UQ) claims made
for generative medical image translation. First, \emph{background air pixels}:
they are trusted and near-zero-error, so pooling them with body pixels
inflates whole-image UQ metrics severalfold---we show a headline localization
score of $0.93$ collapsing to $0.78$ under a within-body protocol, and retract
our own earlier numbers that benefited from it. Second, \emph{dispersion is
not error}: across an equal-budget comparison with TTA and input-noise
baselines, every dispersion-style signal---including ours---is at chance on
T2$\to$T1, where residual error is dominated by fine texture. We then ask
what \emph{does} survive. From the self-consistent recursive diffusion-bridge
sampler (SelfRDB) we harvest two training-free signals: the
\emph{Convergence Residual Field} (CReF), the per-pixel discrepancy between
successive recursive target estimates that the original sampler computes and
discards, and \emph{stochastic-bridge ensemble uncertainty}. Fused into a
pixel-wise trust map with a three-tier QC gate and a six-class input gate
with out-of-distribution rejection, they deliver, under the honest protocol:
AUROC $0.78$ (T1$\to$T2) at zero marginal cost for a single chain,
$0.91$ on real tumor patients, and $0.95$ \emph{cross-institution} on
never-seen subjects. CReF-triggered routing recovers most of full-budget TTA
quality at a fraction of its cost. A dedicated reproducibility analysis
explains why officially released weights collapse to ${\sim}15$\,dB on
independently prepared data (paper value 31.56\,dB), and shows the trust map
flagging the collapse globally without ground truth. We release all negative
results---the protocol trap, the texture-dominated failure boundary, the
absence of tumor localization, the miscalibration of image-level gates---as a
boundary map for the field, and recommend UQ claims in image translation be
evaluated within the body, against cheap baselines, and per (model, direction)
pair.
\end{abstract}"""
s = s[:i0] + new_abs + s[i1:]

# 3) §4.3 改名：失效边界
old3 = "\\subsection{Trust localization and UQ baselines}"
assert old3 in s, "uq subsection"
s = s.replace(old3,
              "\\subsection{Trust localization, UQ baselines, and the failure boundary}")

io.open(fp, "w", encoding="utf-8").write(s)
print("v0.4 patch 1 done: title + abstract + subsection rename")
