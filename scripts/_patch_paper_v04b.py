# -*- coding: utf-8 -*-
"""论文 v0.4 patch 2：复现危机章 + 阴性结果进攻化 + 图像级 QC 劝退。"""
import io

fp = r"D:\MedTranslateProject\docs\latex\main.tex"
s = io.open(fp, encoding="utf-8").read()

# ---- 1) 新增 §4.2 The Reproducibility Crisis（含 15dB 崩盘甄别 + 信任响应）----
anchor = "\\subsection{Translation quality}"
assert anchor in s, "repro anchor"
crisis = r"""\subsection{The reproducibility crisis}
\label{sec:exp-repro}
\paragraph{The collapse.}
Officially released IXI weights reproduce only ${\sim}15$\,dB on independently
prepared IXI data---against the paper's 31.56\,dB (Table~\ref{tab:main}; in
our runs the official T1$\to$T2 weights average $12.98$\,dB within the body).
A systematic exclusion study over the plausible pre-processing degrees of
freedom (rigid misalignment, slice $z$-shift, intensity windowing, cropping,
orientation) traces the gap to pre-processing details the original authors do
not publish; the companion SynDiff release is independently reported to have
the same problem. We emphasize what this means: \emph{a reader who prepares
the data in any reasonable way cannot reproduce the paper}, and the failure
mode is invisible---the official outputs are structurally coherent images
with a global intensity offset (NCC ${\sim}0.9$), not noise.
\paragraph{What the trust layer does and does not see in a collapse.}
A natural hope is that sampler-native signals would ``know'' the base model
is failing. We tested this directly: on the collapsed official model the
trust map does \emph{not} fire---mean trust is $94.2/100$, low-trust ratio
$5.9\%$ (lower than for our healthy self-trained model), and all 20 test
slices receive PASS. The reason is instructive: the collapse is
\emph{photometric}, not structural, and the sampler is perfectly
self-consistent about its (wrong-intensity) output---CReF and dispersion are
correctly reporting that \emph{structure} is reliable, while PSNR punishes
the intensity offset. We conclude that sampler-native trust evidence covers
structural hallucination but is \emph{blind to photometric failure}; a
deployment needs an additional intensity-consistency check against the source
modality, which our input gate does not yet provide (a slice-level
source--output linear-consistency test is straightforward future work).

"""
s = s.replace(anchor, crisis + anchor, 1)

# ---- 2) 图像级 QC：明确劝退全局标量 ----
old = ("We therefore present the three-tier gate as a\n"
       "\\emph{workflow prioritization} device, not a quality filter, ship per-deployment\n"
       "threshold re-tuning as a first-class operation, and state plainly that\n"
       "image-level QC remains unsolved at this data scale.")
assert old in s, "qc reco"
new = ("We therefore draw a stronger conclusion than ``the thresholds need\n"
       "re-tuning'': \emph{global scalar trust scores should not be used as QC\n"
       "gates for medical image translation at all}. Slices differ so much in\n"
       "anatomy that no single body-mean score can separate good from bad\n"
       "across a cohort. We recommend the community standardize on\n"
       "pixel-/region-level trust evidence (the trust map, low-trust region\n"
       "lists) for human review, and treat any image-level gate---ours\n"
       "included---as a triage heuristic at best.")
s = s.replace(old, new)

# ---- 3) 肿瘤盲区：进攻性分析句 ----
old2 = ("Flagged regions indicate where the model is\n"
        "unsure about its synthesis, not where pathology is.")
assert old2 in s, "tumor sentence"
new2 = ("Flagged regions indicate where the model is\n"
        "unsure about its synthesis, not where pathology is. The lesson goes\n"
        "beyond our method: pixel-variance-style UQ measures the \\emph{variance\n"
        "of the synthesis process}, not the \\emph{clinical salience} of the\n"
        "content---a model can be highly confident (and highly consistent) about\n"
        "a pathologically wrong synthesis. Any UQ layer for clinical use must be\n"
        "audited against pathology annotations, not only against error maps.")
s = s.replace(old2, new2)

io.open(fp, "w", encoding="utf-8").write(s)
print("patch 2 done: reproducibility crisis + QC retreat + tumor lesson")
