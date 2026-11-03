# -*- coding: utf-8 -*-
"""论文 v0.4 patch 3：自训模型 UQ 主表 + 官方权重对照 + 失效边界重定位 + 摘要数字。"""
import csv
import io

import numpy as np

ROOT = r"D:\MedTranslateProject"
fp = ROOT + r"\docs\latex\main.tex"
s = io.open(fp, encoding="utf-8").read()

# gradient T2→T1 中位分离（前面漏算）
rows = list(csv.DictReader(open(ROOT + r"\results\2026-11-04_uq_baselines_T2_T1\per_case.csv".replace("per_case", "per_slice"), encoding="utf-8")))
g = np.median([float(r["decile_sep"]) for r in rows if r["config"] == "gradient"])

# ---- 1) Table 2 整体替换（自训模型主表 + 官方权重对照行）----
i0 = s.index("\\begin{table}[t]\n\\centering\\small\n\\caption{Uncertainty-quality comparison")
i1 = s.index("\\end{table}", i0) + len("\\end{table}")
new_tab = r"""\begin{table}[t]
\centering\small
\caption{Uncertainty-quality comparison under the within-body protocol
(IXI test, 60 slices per direction, seed 2026). \textbf{Top:} self-trained
128$\times$128 models (the quantitative backbone); all dispersion baselines
use the same forward-pass budget as our ensemble (40 generator calls); CReF
(N=1) uses a single chain (10 calls, \emph{zero extra cost}). Sep is the
median-across-slices ratio of mean error between least- and most-trusted
deciles; for the self-trained T1$\to$T2 model the most-trusted decile carries
near-zero error, so Sep is an order-of-magnitude statement rather than a
precise value. \textbf{Bottom:} the same protocol on the official weights,
where the signal ranking changes and T2$\to$T1 fails entirely.}
\label{tab:uq}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{T1$\to$T2 (self)} & \multicolumn{3}{c}{T2$\to$T1 (self)} \\
Signal & AUROC & $\Delta$PSNR@20\% & Sep & AUROC & $\Delta$PSNR@20\% & Sep \\
\midrule
Random (control) & 0.498 & $\approx$0 & 1.0 & 0.501 & $\approx$0 & 1.0 \\
Gradient & 0.225 & $-0.45$ & 0.54 & 0.475 & $-0.55$ & """ + f"{g:.2f}" + r""" \\
Input-noise dispersion & 0.861 & $+1.47$ & 65 & 0.739 & $+0.97$ & 3.0 \\
TTA dispersion & 0.820 & $+1.33$ & 6.3 & 0.711 & $+0.79$ & 1.9 \\
CReF only (N=1, free) & 0.880 & $+2.28$ & 103 & 0.792 & $+1.51$ & 2.8 \\
Ensemble only (N=4) & 0.897 & $+2.34$ & 159 & 0.806 & $+1.63$ & 3.8 \\
\textbf{Fused (ours)} & \textbf{0.900} & $\mathbf{+2.67}$ & \textbf{182} & \textbf{0.816} & $\mathbf{+1.85}$ & \textbf{3.7} \\
\midrule
\multicolumn{7}{l}{\emph{Same protocol on the official weights (N=4 fused):}} \\
Fused (official) & 0.783 & $+0.37$ & -- & 0.465 & $+0.21$ & -- \\
TTA (official) & \textbf{0.808} & $+0.70$ & -- & 0.482 & $+0.95$ & -- \\
All signals (official, T2$\to$T1) & -- & -- & -- & \multicolumn{3}{l}{$0.465$--$0.505$ (chance)} \\
\bottomrule
\end{tabular}
\end{table}"""
s = s[:i0] + new_tab + s[i1:]

# ---- 2) 表后分析段重写 ----
i0 = s.index("Table~\\ref{tab:uq} gives the honest picture.")
i1 = s.index("\\subsection{Calibration}")
new_par = r"""Table~\ref{tab:uq} carries two messages. \textbf{On self-trained models},
our fused signal wins outright: AUROC $0.900$/$0.816$ (T1$\to$T2/T2$\to$T1)
with $+2.67$/$+1.85$\,dB sparsification gains, ahead of every equal-budget
baseline, and a single CReF chain (zero extra cost) already reaches
$0.880$/$0.792$---most of the value at a quarter of the budget. Fusion is
robust but does not beat the ensemble alone by much; CReF's claim is cost,
not accuracy. \textbf{On the official weights}, the ranking changes and the
T2$\to$T1 direction collapses entirely: every dispersion signal---ours and
all baselines'---is at chance ($0.465$--$0.505$), because that model's
residual error is dominated by fine texture that no dispersion measure can
track. \emph{The same direction, the same protocol, opposite conclusions on
two models}: the failure boundary is a property of the (model, direction)
pair, not of the signals, which is why we recommend validating uncertainty
signals per (model, direction) before any deployment (\S\ref{sec:discussion}).
A CReF-triggered routing scheme (uncertain regions re-examined by TTA) was
tested to recover full-budget TTA quality at lower cost; on the self-trained
model the curve is flat---pure CReF ($0.880$ at budget 10) already dominates
full TTA ($0.820$ at budget 50)---so routing adds nothing there, a further
confirmation that which signal wins is model-dependent.

"""
s = s[:i0] + new_par + s[i1:]

# ---- 3) 失效边界小节引用 official（§4.3 已含），修正 §4.7 边界(6) ----
old = "(6)~\\emph{T2$\\to$T1 signal failure}\n(\\S\\ref{sec:exp-uq}): all dispersion signals at chance when residual error is\ntexture-dominated."
if old not in s:
    import re
    m = re.search(r"\(6\)~\\emph\{T2\$\\to\$T1 signal failure\}.*?texture-dominated\.", s, re.S)
    old = m.group(0)
new = ("(6)~\\emph{T2$\\to$T1 signal failure on the official weights}\n"
       "(\\S\\ref{sec:exp-uq}): all dispersion signals at chance when that model's\n"
       "residual error is texture-dominated---while the same direction works on our\n"
       "self-trained model ($0.816$), isolating the failure to the model, not the task.")
s = s.replace(old, new)

# ---- 4) 摘要与贡献点数字更新 ----
s = s.replace("AUROC $0.78$ (T1$\\to$T2) at zero marginal cost for a single chain,",
              "AUROC $0.90$ (T1$\\to$T2, self-trained; $0.88$ for a single zero-cost chain),")
s = s.replace("AUROC\n        $0.78$ (T1$\\to$T2, self-trained) and $0.91$ (real tumor patients,\n"
              "        official weights), with sparsification gains up to $+3.1$\\,dB; on\n"
              "        T2$\\to$T1 no dispersion signal---ours or baselines'---tracks error\n"
              "        (\\S\\ref{sec:exp}).",
              "AUROC\n        $0.90$/$0.82$ (self-trained T1$\\to$T2/T2$\\to$T1), beating all\n"
        "        equal-budget baselines, $0.91$ on real tumor patients, and $0.95$\n"
        "        cross-institution; on the \\emph{official} weights the same protocol\n"
        "        flips the ranking and blinds T2$\\to$T1 entirely (\\S\\ref{sec:exp}).")

# ---- 5) 校准一节的官方口径相关系数说明（切片分数已用自训模型重跑，数字回填待结果）----
io.open(fp, "w", encoding="utf-8").write(s)
print("patch 3 done")
