# -*- coding: utf-8 -*-
"""论文 v0.7 patch B：§4.5 改名 + 指标定义段 + ImplDetails + 复现排查表 + 两层架构 + RW 对照表（双语）。"""
import io

ZH = r"D:\MedTranslateProject\docs\latex\main_zh.tex"
EN = r"D:\MedTranslateProject\docs\latex\main.tex"
RW_ZH = r"D:\MedTranslateProject\docs\latex\related_work_zh.tex"
RW_EN = r"D:\MedTranslateProject\docs\latex\related_work.tex"


def splice(path, start, end, new, tag="auto", required=True):
    s = io.open(path, encoding="utf-8").read()
    if new[:60] in s:
        print(f"{tag}: already applied"); return
    try:
        i0 = s.index(start)
        i1 = s.index(end, i0) + len(end)
    except ValueError:
        if required:
            raise
        print(f"{tag}: markers not found, skipped"); return
    s = s[:i0] + new + s[i1:]
    io.open(path, "w", encoding="utf-8").write(s)
    print(f"{tag}: spliced")


# ============ ZH ============
# §4.5 改名
splice(ZH, "\\subsection{校准}", "\\subsection{校准}",
       "\\subsection{误差排序与十分位分离}", "ZH-4.5-rename")

# §4.1 指标定义段（插在伦理段前）
splice(ZH, "\\paragraph{伦理：}",
       "三个数据集均为公开研究数据集；全部处理本地完成；任何输出不用于临床决策。",
       """\\paragraph{指标与掩码定义（冻结于 definitions v1）。}
全部信任指标在体掩码内逐片计算：$M=\\{\\text{源图}>0.02\\}$（源图为逐片 min-max 归一化后的
$[0,1]$ 图；无形态学后处理——简化协议，阈值 0.01/0.05 的敏感性见 \\S4.9）。正类=该片体区内
绝对误差前 10\\% 像素；分数 $=1-T(p)$；AUROC 逐片计算后宏平均（另有 pixel-pooled 与
subject-macro 口径，\\S4.4）；AUPRC 同正类定义并报告实际正类比例。$\\Delta$PSNR@k\\% =
PSNR$_{\\text{retained}(100-k)\\%}$ $-$ PSNR$_{\\text{full}}$（mean-norm，data\\_range=掩码内
归一化 GT 最大值）。十分箱分离 Sep 为比值、ΔMAE 为绝对差——分母趋零时 Sep 只作数量级陈述
（\\S4.5）。AUSE 为模型与 oracle 风险-覆盖曲线之差的逐片宏平均积分。
\\paragraph{伦理。}
三个数据集均为公开研究数据集；全部处理本地完成；任何输出不用于临床决策。""", "ZH-4.1-defs")

# §4.2 复现排查表（插在"要强调的是这意味着什么"句后）
splice(ZH, "未找到的旧锚点_跳过", "也未找到_跳过。",
       """要强调的是这意味着什么：\\emph{在本文系统测试的独立重建预处理范围内
（对齐、$z$ 偏移、强度窗、裁剪、方向），未能复现已发布权重的报告性能}，
且失效形态不可见——官方输出是平滑、貌似合理的图像而非噪声，但结构上是错的
（体区 NCC 0.41，见下）。精确原因尚未解决：checkpoint 版本不一致、训练代码与发布权重
不一致、依赖库差异等假设均在排查表内（docs/18 复现排查表区分"已核查/未核查"与证据），
在我们获得原作者原始制备代码或确认之前，不进一步锁定归因。""", "ZH-4.2-table", required=False)

# §5 两层架构（插在局限性前）
splice(ZH, "\\paragraph{局限性。}",
       "(1)~\\TrustMap{}标记\\emph{不可靠}区域而非\\emph{错误}区域：",
       """\\paragraph{两层安全架构的定位。}
本文的发现把系统拆成两层：\\emph{输入完整性监测}（域外检测、强度分布异常、模糊/噪声/
错位——输入侧 Mahalanobis 距离已覆盖其中部分）与\\emph{采样器原生信任估计}
（CReF+集成分歧的像素级排序）；两者之外，自信的结构性幻觉与图像级质量退化
必须由外部验证与人工复核兜底。TrustBridge 不是完整的医学图像安全系统，
而是可信工作流中的内部信号评估层。
\\paragraph{局限性。}
(1)~\\TrustMap{}标记\\emph{不可靠}区域而非\\emph{错误}区域：""", "ZH-5-two-layer", required=False)

# ============ EN ============
splice(EN, "\\subsection{Calibration}", "\\subsection{Calibration}",
       "\\subsection{Error ranking and decile separation}", "EN-4.5-rename")

splice(EN, "\\paragraph{Ethics.}",
       "All three datasets are public research datasets; all processing is local; no\noutput is used for clinical decisions.",
       """\\paragraph{Metric and mask definitions (frozen as definitions v1).}
All trust metrics are computed per slice within a body mask:
$M=\\{\\text{source}>0.02\\}$ on the per-slice min-max-normalized $[0,1]$ source
image (no morphological post-processing --- a simplified protocol; threshold
sensitivity at 0.01/0.05 in \\S\\ref{sec:exp-stress}). The positive class is the
top-10\\% absolute-error pixels within the body of each slice; the score is
$1-T(p)$; AUROC is computed per slice and macro-averaged (pixel-pooled and
subject-macro variants in \\S\\ref{sec:exp-uq}); AUPRC uses the same positive
definition with the realized positive rate reported. $\\Delta$PSNR@k\\% =
PSNR$_{\\text{retained}(100-k)\\%}$ $-$ PSNR$_{\\text{full}}$ (mean-norm,
data\\_range = masked normalized GT maximum). The decile separation Sep is a
ratio and ΔMAE its absolute-difference counterpart --- when the denominator
approaches zero, Sep is an order-of-magnitude statement only (\\S4.5). AUSE is
the per-slice macro-averaged integral of the model-minus-oracle risk--coverage
difference.
\\paragraph{Ethics.}
All three datasets are public research datasets; all processing is local; no
output is used for clinical decisions.""", "EN-4.1-defs")

# EN-4.2-attrib: already applied via Edit tool
# EN-4.2-attrib end

splice(EN, "\\paragraph{Limitations.}",
       "(1)~The trust map flags \\emph{unreliable} regions, not \\emph{wrong} ones:",
       """\\paragraph{Positioning: a two-layer safety architecture.}
Our findings split the system into two layers: an \\emph{input integrity
monitor} (out-of-distribution detection, intensity anomalies, blur/noise/
misalignment --- partially covered by the input-side Mahalanobis distance) and
a \\emph{sampler-native trust estimator} (pixel-wise ranking by CReF and
ensemble divergence); beyond both, confident structural hallucination and
image-level quality degradation must be caught by external validation and human
review. TrustBridge is not a complete medical-imaging safety system but an
internal-signal evaluation layer within a trustworthy workflow.
\\paragraph{Limitations.}
(1)~The trust map flags \\emph{unreliable} regions, not \\emph{wrong} ones:""", "EN-5-two-layer")

# ============ RW 对照表 ============
RW_TABLE_ZH = """

\\paragraph{方法对照。}
表~\\ref{tab:rw} 按三个维度对比（以各原论文的真实实现为准填写）。
\\begin{table}[h]
\\centering\\small
\\caption{不确定性/幻觉评估方法对照（"额外前向"指相对单次基础推理的生成器前传次数）。
\\TrustBridge{} 一行的"无额外前传"限定于当前 SelfRDB 实现——其递归结构为常规 DDPM 所无。}
\\label{tab:rw}
\\begin{tabular}{lcccc}
\\toprule
方法 & 需要训练 & 额外模型 & 额外前向 & 输出粒度 \\\\
\\midrule
MC-Dropout~\\cite{gal2016mcdropout} & 需（启用 dropout 重训） & 否 & 多次 & 像素/图像 \\\\
深度集成~\\cite{lakshminarayanan2017ensembles} & 需（多模型） & 是 & 多次 & 像素/图像 \\\\
TTA~\\cite{ayhan2018tta} & 否 & 否 & 多次 & 像素/图像 \\\\
扩散集成~\\cite{wolleb2022diffusionensembles,osuna2024microscopy} & 否（复用模型） & 否 & 多次 & 像素/图像 \\\\
Hallucination Index~\\cite{tivnan2024hallucinationindex} & 视实现 & 视实现 & 多次 & 图像级 \\\\
\\TrustBridge{}（本文） & 否 & 否 & \\textbf{无额外}（单链） & 像素级 \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
RW_TABLE_EN = """

\\paragraph{Method comparison.}
Table~\\ref{tab:rw} compares along three dimensions, filled from each original
paper's actual implementation.
\\begin{table}[h]
\\centering\\small
\\caption{Uncertainty/hallucination assessment methods ("extra forwards" counts
generator calls beyond a single base inference). The "no extra" entry of
\\TrustBridge{} is specific to the current SelfRDB implementation --- its
recursion structure does not exist in vanilla DDPMs.}
\\label{tab:rw}
\\begin{tabular}{lcccc}
\\toprule
Method & Training & Extra model & Extra forwards & Granularity \\\\
\\midrule
MC-Dropout~\\cite{gal2016mcdropout} & yes (retrain w/ dropout) & no & multiple & pixel/image \\\\
Deep ensembles~\\cite{lakshminarayanan2017ensembles} & yes (multiple models) & yes & multiple & pixel/image \\\\
TTA~\\cite{ayhan2018tta} & no & no & multiple & pixel/image \\\\
Diffusion ensembles~\\cite{wolleb2022diffusionensembles,osuna2024microscopy} & no (reuse) & no & multiple & pixel/image \\\\
Hallucination Index~\\cite{tivnan2024hallucinationindex} & depends & depends & multiple & image-level \\\\
\\TrustBridge{} (ours) & no & no & \\textbf{none} (single chain) & pixel-level \\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
splice(RW_ZH, "我们将其准确定位为", "采样器内部收敛不稳定性的代理信号。", RW_TABLE_ZH, "ZH-RW-table", required=False)
splice(RW_EN, "We position it precisely as", "a proxy of sampler-internal convergence instability", RW_TABLE_EN, "EN-RW-table", required=False)

print("patch B complete")
