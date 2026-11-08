# -*- coding: utf-8 -*-
"""EN figure insertion (ZH done in v06b first run)."""
import io

EN = r"D:\MedTranslateProject\docs\latex\main.tex"


def patch(path, pairs, tag):
    s = io.open(path, encoding="utf-8").read()
    for i, (old, new) in enumerate(pairs):
        assert old in s, f"[{tag}] anchor #{i} not found: {old[:70]!r}"
        s = s.replace(old, new, 1)
    io.open(path, "w", encoding="utf-8").write(s)
    print(f"{tag}: {len(pairs)} edits applied")


FIG_OVERVIEW_EN = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{figures/fig1_overview_v2.pdf}
\caption{Method overview. Left: the frozen SelfRDB translation model (forward
bridge and self-consistent recursive reverse chain); right: the
trust-quantification layer --- the convergence residual field (signal 1) and
multi-chain divergence (signal 2) are harvested from the recursion and fused
into a pixel-wise trust map. Both signals are already computed by the sampler;
harvesting them adds no model, no training and no labels.}
\label{fig:overview}
\end{figure}
"""

FIG_CASE_EN = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{figures/fig3_trust_example.pdf}
\caption{Per-slice six-panel visualization (T2$\to$T1, self-trained): source,
synthesis, CReF, ensemble divergence, trust map, and the ground-truth absolute
error (evaluation only). The spatial correspondence between the trust map and
the error map is the core qualitative evidence: bright = trustworthy,
dark = suspicious. The pattern is consistent across the 60 test slices.}
\label{fig:case}
\end{figure}
"""

FIG_COLLAPSE_EN = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.8\linewidth]{figures/fig_collapse_vs_healthy.pdf}
\caption{Image-level trust distribution: collapsed base model vs.\ healthy
model (20 test slices each, seed 2026). The structurally collapsed official
weights (12.98\,dB) receive HIGHER trust (mean 94.2, all PASS) than the
healthy self-trained model (90.1) --- variance-type signals cannot see the
failure.}
\label{fig:collapse}
\end{figure}
"""

FIG_DECILE_EN = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/fig_body_decile.pdf}
\caption{Within-body reliability: mean absolute error per trust decile
(60 test slices per direction, seed 2026; error bars = IQR). Both directions
are near-monotone; the within-body separation is far smaller than the
whole-image number (the retracted 35--73$\times$ was dominated by air/body
contrast and is not used as a method metric). The top three bins carry
near-zero error, where Spearman is insensitive.}
\label{fig:decile}
\end{figure}
"""

FIG_TUMOR_EN = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{figures/fig6_tumor.pdf}
\caption{Lesion blind spot (corrected, BraTS2018, 60 tumor-bearing slices,
within-body protocol). (a) mean trust inside/outside tumor; (b) mean absolute
error --- 1.60$\times$ inside; (c) within-body AUROC of low-trust-indicates-tumor
--- mean 0.51, chance level (red line); (d--f) case slices: FLAIR (tumor
bright), trust map + tumor contour (tumor is HIGH-trust), translation. The
model confidently interpolates the homogeneous bright region into wrong
content (bias-type hallucination), invisible to variance-type signals.}
\label{fig:tumor}
\end{figure}
"""

en_pairs = [
    ("""prime candidates for hallucinated
content. The signal is epistemic in the sense
of~\\cite{kendall2017uncertainties} and costs nothing: the recursion runs
anyway.""",
     """prime candidates for hallucinated
content. The signal is epistemic in the sense
of~\\cite{kendall2017uncertainties} and costs nothing: the recursion runs
anyway. Figure~\\ref{fig:overview} relates both signals to the sampling chain.
""" + FIG_OVERVIEW_EN),
    ("""never seen its base model fail cannot be trusted to recognize failure.""",
     """never seen its base model fail cannot be trusted to recognize failure
(Figure~\\ref{fig:collapse}).
""" + FIG_COLLAPSE_EN),
    ("""confirmation that which signal wins is model-dependent.

\\subsection{Calibration}""",
     """confirmation that which signal wins is model-dependent.
Figure~\\ref{fig:case} shows the per-slice six-panel visualization: the spatial
correspondence between the trust map and the true error underlies all
quantitative claims.

\\subsection{Calibration}
""" + FIG_CASE_EN),
    ("""a \\emph{ranking within the body}, not an absolute error magnitude.""",
     """a \\emph{ranking within the body}, not an absolute error magnitude
(Figure~\\ref{fig:decile}).
""" + FIG_DECILE_EN),
    ("""audited against pathology annotations, not only against error maps.""",
     """audited against pathology annotations, not only against error maps
(Figure~\\ref{fig:tumor}).
""" + FIG_TUMOR_EN),
]

patch(EN, en_pairs, "EN-figs")
