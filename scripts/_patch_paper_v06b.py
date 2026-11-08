# -*- coding: utf-8 -*-
"""论文 v0.6 patch 2：接入 5 张图（figure 环境）+ 文中引用句，中英双稿。"""
import io

ZH = r"D:\MedTranslateProject\docs\latex\main_zh.tex"
EN = r"D:\MedTranslateProject\docs\latex\main.tex"


def patch(path, pairs, tag):
    s = io.open(path, encoding="utf-8").read()
    for i, (old, new) in enumerate(pairs):
        assert old in s, f"[{tag}] anchor #{i} not found: {old[:70]!r}"
        s = s.replace(old, new, 1)
    io.open(path, "w", encoding="utf-8").write(s)
    print(f"{tag}: {len(pairs)} edits applied")


FIG_OVERVIEW = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{figures/fig1_overview_v2.pdf}
\caption{方法总览。左：冻结的 SelfRDB 翻译模型（前向桥与含自洽递归的反向链）；右：
可信度量化层——从递归过程提取收敛残差场（信号一）与多链分歧（信号二），融合为逐像素信任图。
两个信号均由采样器本身计算，收割它们不引入任何模型、训练或标注。}
\label{fig:overview}
\end{figure}
"""

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

FIG_CASE_ZH = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{figures/fig3_trust_example.pdf}
\caption{典型切片六联可视化（T2$\to$T1，自训模型）：源图、合成图、CReF、集成分歧、
信任图与真实绝对误差（仅评估用）。信任图与误差图的空间对应关系是本文定性证据的核心：
亮=可信、暗=可疑。测试集 60 片中逐片模式一致。}
\label{fig:case}
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

FIG_COLLAPSE_ZH = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.8\linewidth]{figures/fig_collapse_vs_healthy.pdf}
\caption{崩塌基座模型 vs 健康模型的图像级信任分布（各 20 片测试切片，seed 2026）。
结构性崩塌的官方权重（12.98 dB）获得的信任（均值 94.2，全部 PASS）反而高于健康的
自训模型（90.1）——方差型信号对"自信地犯错"失明。}
\label{fig:collapse}
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

FIG_DECILE_ZH = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.95\linewidth]{figures/fig_body_decile.pdf}
\caption{体区口径可靠性：按信任十分箱后的各箱平均绝对误差（每方向 60 测试切片，
seed 2026，误差棒为四分位距）。两方向均近单调递减；体区口径下分离幅度远小于
整图口径（历史值 35--73$\times$ 系空气/体区对比主导，已撤回，不作为方法指标）。
最高三箱误差接近零，Spearman 在零值段不敏感。}
\label{fig:decile}
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

FIG_TUMOR_ZH = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{figures/fig6_tumor.pdf}
\caption{病灶盲区（修正版，BraTS2018，60 个含瘤切片，体掩码内口径）。
(a) 瘤内/体（非瘤）逐片平均信任；(b) 逐片平均绝对误差——瘤内为瘤外的 1.60 倍；
(c) 体掩码内低信任指示肿瘤的 AUROC 分布——均值 0.51，等于随机水平（红线）；
(d--f) 案例切片：FLAIR（瘤区亮）、信任图+瘤轮廓（瘤区为高信任）、翻译结果。
模型在均质亮区自洽地插值出错误内容（偏差型幻觉），方差型信号不可见。}
\label{fig:tumor}
\end{figure}
"""

FIG_TUMOR_EN = r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.98\linewidth]{figures/fig6_tumor.pdf}
\caption{Lesion blind spot (corrected, BraTS2018, 60 tumor-bearing slices,
within-body protocol). (a) mean trust inside/outside tumor; (b) mean absolute
error --- 1.60$\times$ inside; (c) within-body AUROC of low-trust-indicates-
tumor --- mean 0.51, chance level (red line); (d--f) case slices: FLAIR (tumor
bright), trust map + tumor contour (tumor is HIGH-trust), translation. The
model confidently interpolates the homogeneous bright region into wrong
content (bias-type hallucination), invisible to variance-type signals.}
\label{fig:tumor}
\end{figure}
"""

zh_pairs = [
    # fig1：放在 §3.2 CReF 小节末尾（直觉段之后）
    ("""直觉：生成器反复修正仍无法自洽的像素，正是其学到的
先验证据不足之处——幻觉候选区。该信号在~\\cite{kendall2017uncertainties} 意义下属于认知性
不确定性，且成本为零：递归本来就要算。""",
     """直觉：生成器反复修正仍无法自洽的像素，正是其学到的
先验证据不足之处——幻觉候选区。该信号在~\\cite{kendall2017uncertainties} 意义下属于认知性
不确定性，且成本为零：递归本来就要算。两类信号与采样链的关系见图~\\ref{fig:overview}。
""" + FIG_OVERVIEW),
    # fig_collapse：§4.2 复现危机段末
    ("""部署教训：\\emph{免训练内部信号无法察觉"自信地犯错"的基座模型}。
它们补充、但不能替代对真值的外部验证；从未见过基座模型失败的守卫层，不可被信任去识别失败。""",
     """部署教训：\\emph{免训练内部信号无法察觉"自信地犯错"的基座模型}。
它们补充、但不能替代对真值的外部验证；从未见过基座模型失败的守卫层，不可被信任去识别失败
（图~\\ref{fig:collapse}）。
""" + FIG_COLLAPSE_ZH),
    # fig_case：§4.4 末尾
    ("""进一步确认信号排序的模型相关性。

\\subsection{校准}""",
     """进一步确认信号排序的模型相关性。图~\\ref{fig:case} 给出典型切片的六联可视化：
信任图与真实误差的空间对应关系是全部定量结论的直觉基础。

\\subsection{校准}
""" + FIG_CASE_ZH),
    # fig_decile：§4.5 校准
    ("""\\TrustMap{}应作为\\emph{体区内的排序}使用，
而非绝对误差量级。""",
     """\\TrustMap{}应作为\\emph{体区内的排序}使用，
而非绝对误差量级（图~\\ref{fig:decile}）。
""" + FIG_DECILE_ZH),
    # fig_tumor：§4.8 肿瘤
    ("""任何面向临床的 UQ 层都必须对照
病理标注审计，而不能只对照误差图。""",
     """任何面向临床的 UQ 层都必须对照
病理标注审计，而不能只对照误差图（图~\\ref{fig:tumor}）。
""" + FIG_TUMOR_ZH),
]

en_pairs = [
    ("""直觉PLACEHOLDER""", """"""),
]

en_pairs = [
    # fig1 at end of §3.2
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
    ("""further confirmation that which signal wins is model-dependent.

\\subsection{Calibration}""",
     """further confirmation that which signal wins is model-dependent.
Figure~\\ref{fig:case} shows the per-slice six-panel visualization: the spatial
correspondence between the trust map and the true error underlies all
quantitative claims.

\\subsection{Calibration}
""" + FIG_CASE_EN),
    ("""\\TrustMap{} should be read as a \\emph{ranking within the body}, not an
absolute error magnitude.""",
     """\\TrustMap{} should be read as a \\emph{ranking within the body}, not an
absolute error magnitude (Figure~\\ref{fig:decile}).
""" + FIG_DECILE_EN),
    ("""audited against pathology annotations, not only against error maps.""",
     """audited against pathology annotations, not only against error maps
(Figure~\\ref{fig:tumor}).
""" + FIG_TUMOR_EN),
]

patch(ZH, zh_pairs, "ZH-figs")
patch(EN, en_pairs, "EN-figs")
print("patch 2 complete")
