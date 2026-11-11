# -*- coding: utf-8 -*-
"""v0.8.3 EN 正文补丁 II：§4.4 敏感性段 + SynthRAD 跨解剖小节。"""
import io

P = r"D:\MedTranslateProject\docs\latex\main.tex"
s = io.open(P, encoding="utf-8").read()


def rep(old, new, tag):
    global s
    assert old in s, f"[{tag}] NOT FOUND: {old[:80]!r}"
    assert s.count(old) == 1, f"[{tag}] NOT UNIQUE: {old[:80]!r}"
    s = s.replace(old, new, 1)


# ---------- 4. §4.4 敏感性段（插在 fig:case 引用段之前） ----------
rep(r"""confirmation that which signal wins is model-dependent.
Figure~\ref{fig:case} shows the per-slice six-panel visualization""",
    r"""confirmation that which signal wins is model-dependent.
\paragraph{Sensitivity to the frozen operators.}
Because the fusion weight, the capture window $K_0$, and the
99th-percentile normalization are all fixed a priori, we verify that no
conclusion depends on their precise values. Sweeping the fusion weight
$w_1$ (the CReF share) at equal budget moves within-body AUROC by at most
$0.005$ across $w_1\in[0.2,0.6]$ ($0.826/0.827/0.822$ at $w_1{=}0.2/0.4/0.6$
on T1$\to$T2; $0.738/0.744/0.739$ on T2$\to$T1), with both endpoints lower
($0.812$ ensemble-only, $0.796$ CReF-only on T1$\to$T2). The pre-specified
$w_1{=}0.6$ sits on this plateau; we report the sweep as a sensitivity
analysis, not a selection. Varying the capture window
$K_0\in\{1,2,3,5,10\}$ changes AUROC by at most $0.004$
($0.820/0.822/0.822/0.821/0.818$; note $K_0{=}1$ is semantically
degenerate---the residual becomes the generator output itself, as
$\tilde x_0^{(0)}\equiv0$---yet the metric is unchanged). On cross-anatomy
pelvis data neither operator is innocuous: the same $K_0$ sweep spans
$0.628$--$0.743$ and the fusion-weight ranking reverses
(\S\ref{sec:exp-synthrad}). The per-slice Spearman correlation between CReF
and ensemble divergence is $0.806\pm0.054$ (T1$\to$T2) versus
$0.475\pm0.077$ (T2$\to$T1): the two signals are nearly redundant where
translation is easy and substantially complementary where it is hard,
consistent with fusion helping mainly on T2$\to$T1 (Table~\ref{tab:uq}).
Against the same top-10\%-error positive class, AUPRC is
$0.348\pm0.051$/$0.271\pm0.087$ (random ranking $\approx0.10$) and AUSE
$0.014\pm0.004$/$0.031\pm0.013$ (T1$\to$T2/T2$\to$T1).
Figure~\ref{fig:case} shows the per-slice six-panel visualization""",
    "sensitivity-para")

# ---------- 5. SynthRAD 小节（插在 Discussion 之前） ----------
rep(r"""not an artifact of having seen the patients.

\section{Discussion}""",
    r"""not an artifact of having seen the patients.

\subsection{Cross-anatomy transfer: pelvis CT (SynthRAD)}
\label{sec:exp-synthrad}
The stronger test of sampler-native trust is not a new institution but a
new anatomy and a new target modality. We run the identical frozen pipeline
on the self-trained pelvis bridges (T1w$\to$CT: 160 slices, 2 patients;
T2w$\to$CT: 80 slices, 1 patient; all prepared SynthRAD slices, including
the designated validation split, with no configuration re-tuned).
Slice-level results: T1w$\to$CT reaches within-body AUROC $0.666\pm0.061$
with AUPRC $0.285\pm0.078$ (positive rate $0.10$), median decile separation
$1.39$, and AUSE $0.037$ --- degraded relative to brain but still
informative --- whereas T2w$\to$CT gives AUROC $0.591\pm0.070$, AUPRC
$0.155$, and median decile separation $0.86$: \emph{indistinguishable from
chance at the slice level}, on the hardest translation in our suite (body
PSNR $15.3$\,dB versus $18.6$\,dB).
Two honest observations accompany these numbers. First, the frozen
operators are not anatomy-free: sweeping $K_0$ from 1 to 10 moves T1w$\to$CT
AUROC from $0.628$ to $0.743$ (the brain-tuned $K_0{=}3$ is suboptimal
here), and the fusion-weight ranking \emph{reverses} --- ensemble-only
attains $0.698$ while CReF-only drops to $0.575$, so on this (model,
direction) pair the convergence residual is the weaker error carrier.
Second, with 1--2 patients these are slice-level statistics without
patient-level variance, and mask-threshold sensitivity is larger than on
brain ($0.639/0.666/0.718$ at thresholds $0.01/0.02/0.05$). The pattern
extends our failure-boundary thesis: whether dispersion carries error
information is a property of the (model, direction, anatomy) triple, and
must be discovered by evaluation, not assumed --- which is exactly why the
per-deployment validation of \S\ref{sec:discussion} comes before any use.

\section{Discussion}""", "synthrad-sec")

io.open(P, "w", encoding="utf-8").write(s)
print("EN patches 4-5 OK")
