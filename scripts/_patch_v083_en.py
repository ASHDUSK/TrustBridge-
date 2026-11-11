# -*- coding: utf-8 -*-
"""v0.8.3 EN 正文补丁：数据集勘误 + 阈值敏感性 + 参数表 + §4.4 敏感性段 + SynthRAD 小节。"""
import io

P = r"D:\MedTranslateProject\docs\latex\main.tex"
s = io.open(P, encoding="utf-8").read()


def rep(old, new, tag):
    global s
    assert old in s, f"[{tag}] NOT FOUND: {old[:80]!r}"
    assert s.count(old) == 1, f"[{tag}] NOT UNIQUE: {old[:80]!r}"
    s = s.replace(old, new, 1)


# ---------- 1. Setup Datasets: SynthRAD 3 patients ----------
rep(r"""segmentation; source FLAIR. \textbf{SynthRAD2023 pelvis}~\cite{thummerer2023synthrad}:
10 patients, center-matched T1w$\to$CT and T2w$\to$CT, 80 test slices each.""",
    r"""segmentation; source FLAIR. \textbf{SynthRAD2023 pelvis}~\cite{thummerer2023synthrad}:
3 patients (2 for T1w$\to$CT, 1 for T2w$\to$CT; 128$\times$128 slices, 160 and
80 respectively, all prepared slices used) --- a pilot-scale cross-anatomy
probe rather than a population sample (\S\ref{sec:exp-synthrad}).""",
    "datasets")

# ---------- 2. Setup threshold sensitivity ----------
rep(r"""(no morphological post-processing --- a simplified protocol; threshold
sensitivity at 0.01/0.05 in \S\ref{sec:exp-stress}).""",
    r"""(no morphological post-processing --- a simplified protocol). Mask-threshold
sensitivity is minor on brain data (within-body AUROC at thresholds
$0.01/0.02/0.05$: $0.830/0.822/0.817$ for T1$\to$T2 and $0.753/0.739/0.737$
for T2$\to$T1) and larger on cross-anatomy data (\S\ref{sec:exp-synthrad}).""",
    "threshold")

# ---------- 3. ImplDetails 表 ----------
rep(r"""the BraTS FLAIR$\to$T2 evaluation.
\paragraph{Trust-metric protocol.}""",
    r"""the BraTS FLAIR$\to$T2 evaluation.
\paragraph{Implementation and analysis configuration.}
Table~\ref{tab:impl} freezes the single configuration used for every number
in this paper; no parameter below is tuned per experiment, and deviations
from it are reported as sensitivity analyses (\S\ref{sec:exp-uq},
\S\ref{sec:exp-synthrad}).
\begin{table}[t]
\centering\small
\caption{Implementation and analysis configuration (one configuration for
all experiments; inference seed 2026 throughout).}
\label{tab:impl}
\begin{tabular}{p{0.24\linewidth}p{0.68\linewidth}}
\toprule
Component & Setting \\
\midrule
Base model & SelfRDB bridge, 10 diffusion steps ($\beta$: $0.1\to3.0$) \\
Self-trained & 128$\times$128, 30 epochs, batch 8, lr $1.6\cdot10^{-4}$/$10^{-4}$ (G/D), Adam $(0.5,0.9)$, bf16-mixed, seed 42, one RTX 4060 Laptop 8\,GB \\
Sampling & $N{=}4$ independent chains, recursion cap $R_{\max}{=}2$, step-skip 1 \\
Capture window & $K_0{=}3$ data-end bridge steps ($\mathcal T_i{=}\{t\le K_0\}$) \\
Normalization & per-map 99th-percentile clip to $[0,1]$ (fixed operator) \\
Fusion & $T = 1-(0.6\,\mathrm{CReF}_n + 0.4\,U_n)$; weight set a priori \\
Equal budget & 40 generator forwards for all $N{=}4$ dispersion signals \\
Image-level score & Trust Score $=100\times$ body-mean of $T$; PASS/REVIEW gates at 75/55 (demo defaults, uncalibrated) \\
Input gate & 6-class modality head + per-class Mahalanobis ($\tau{=}389$, fit on the gate's own training split) + flat-image rule \\
Statistics & per-slice metric, macro-averaged; patient-level paired bootstrap $B{=}5000$ \\
\bottomrule
\end{tabular}
\end{table}
\paragraph{Trust-metric protocol.}""", "impl")

io.open(P, "w", encoding="utf-8").write(s)
print("EN patches 1-3 OK")
