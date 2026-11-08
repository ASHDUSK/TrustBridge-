# -*- coding: utf-8 -*-
"""论文 v0.6 patch 1：去产品化（门控全删退场）+ 贡献/局限/结论调整，中英双稿同步。"""
import io

ZH = r"D:\MedTranslateProject\docs\latex\main_zh.tex"
EN = r"D:\MedTranslateProject\docs\latex\main.tex"


def patch(path, pairs, tag):
    s = io.open(path, encoding="utf-8").read()
    for i, (old, new) in enumerate(pairs):
        assert old in s, f"[{tag}] anchor #{i} not found: {old[:60]!r}"
        s = s.replace(old, new, 1)
    io.open(path, "w", encoding="utf-8").write(s)
    print(f"{tag}: {len(pairs)} edits applied")


# ================= 中文稿 =================
zh_pairs = [
    # 1) 摘要：门控退场 + 体掩码口径显式
    ("""融合为逐像素\\TrustMap{}并配三级 QC 门控
与带域外拒绝的六类输入门控后，在诚实口径下它们给出：AUROC $0.90$（T1$\\to$T2 自训）与 $0.82$""",
     """融合为逐像素\\TrustMap{}后，在体掩码口径下它们给出：AUROC $0.90$（T1$\\to$T2 自训）与 $0.82$"""),
    # 2) 贡献 1：门控退场
    ("""  \\item 我们提出\\emph{收敛残差场（CReF）}与随机桥集成不确定性作为扩散桥翻译的互补、
        免训练信任信号，融合为带三级 QC 门控的像素级\\TrustMap{}；并配六类输入门控
        （含 Mahalanobis 域外拒绝），在合成\\emph{之前}硬拦截错配或异常输入（\\S3）。""",
     """  \\item 我们提出\\emph{收敛残差场（CReF）}与随机桥集成不确定性作为扩散桥翻译的互补、
        免训练、像素级的可信度证据，融合为\\TrustMap{}（\\S3）。"""),
    # 3) 贡献 3：加回崩塌盲区
    ("""官方权重 T2$\\to$T1 信号失效、语义代理无互补——
        并以此划定方法边界：""",
     """官方权重 T2$\\to$T1 信号失效、语义代理无互补、以及基座模型崩塌时信任层失明——
        并以此划定方法边界："""),
    # 4) §3.4 标题与正文：门控退场
    ("""\\subsection{信任图、信任分与 QC 门控}""",
     """\\subsection{信任图融合}"""),
    ("""$w_2{=}0.4$。图像级\\TrustScore{}为体区均值缩放至 0--100，配三级门控
（PASS/REVIEW/REJECT）；\\S4.6 分析这些门槛在三个队列上实际达到的操作点。逐片报告
（PNG + JSON）归档信任图、各步递归统计与输入门控判定。""",
     """$w_2{=}0.4$。图像级聚合（体区均值）的可行性在 \\S4.6 作为负结果讨论；逐片归档
（信任图与各步递归统计）随开源代码库提供。"""),
    # 5) §3.5 输入门控整段删除
    ("""\\paragraph{输入门控 v2。}
在任何合成发生之前，六类分类器（脑 T1/T2/FLAIR、盆腔 T1w/T2w、盆腔 CT；
体区标准化分位数与梯度特征）核对输入与任务声明源模态是否一致，辅以 Mahalanobis
域外检验（阈值取验证集 99.5 分位）与平坦图硬规则。在 1{,}160 例验证矩阵
（匹配/跨模态/跨解剖/合成 OOD 四类套件）中，\\emph{没有}任何非法输入（0/562）获得放行判定，
合法输入零误拒。脑任务对错配输入硬拒绝；盆腔 CT 任务降级为警示放行——未见患者盆腔 T1w
有 $52.5\\%$ 被误判为脑类，硬拒绝会拦掉一半合法输入（\\S5）。

""", ""),
    # 6) §4.9 压力测试：门控改为"测量仪器"表述
    ("""教学意义：\\emph{输入门控}的 Mahalanobis 距离对模糊（$2.9\\to15.0$）与噪声（$\\to9.3$）""",
     """教学意义：输入侧 Mahalanobis 距离（开源输入防御层的域外评分，此处仅作测量仪器）对模糊（$2.9\\to15.0$）与噪声（$\\to9.3$）"""),
    ("""这里"知道自己不知道"由\\emph{输入门控}
承担而非信任图；错位检测仍未解决（\\S5 讨论）。""",
     """这里"知道自己不知道"由输入侧 Mahalanobis 检测承担而非信任图；错位检测仍未解决（\\S5 讨论）。"""),
    # 7) §4.8 肿瘤：原理性盲区句
    ("""被标记区域指示模型对合成不确定的位置，不是病灶所在。教训超出
本文方法：""",
     """被标记区域指示模型对合成不确定的位置，不是病灶所在。这一边界不是实现缺陷，
而是方差型信号的原理性盲区。教训超出本文方法："""),
    # 8) §5 开头：有效/失效条件段
    ("""\\section{讨论}
\\label{sec:discussion}
\\paragraph{质量与信任校准可分离。}""",
     """\\section{讨论}
\\label{sec:discussion}
\\paragraph{有效与失效条件。}
信任层有效的条件：残差误差呈空间结构、基座模型未崩塌、评估在体区内进行。
失效的条件：纹理主导的误差、偏差型幻觉（病灶）、模型级崩塌（\\S\\ref{sec:exp-repro}）、
图像级聚合（\\S4.6）。带着这张清单，下文逐条展开。
\\paragraph{质量与信任校准可分离。}"""),
    # 9) 局限 (3)(4)(5)(6) 重写
    ("""(3)~输入门控对脑任务硬拒绝、对盆腔 CT 任务仅警示：未见患者盆腔 T1w $52.5\\%$ 被误判为
脑类，硬拒绝会拦掉一半合法输入；Mahalanobis OOD 阈值另被跨扫描仪域移位抬高，
部分结构化合成探针由分类器层而非距离检验兜住。
(4)~图像级 QC 与肿瘤定位在此规模仍未解决；信任图还不响应输入退化——模糊反而抬高它
（\\S\\ref{sec:exp-stress}）——退化输入检测依赖输入门控，小幅错位两者皆盲。
(5)~三级门控阈值需逐部署重调（\\S4.6）。
(6)~全部输出为研究演示，不得指导临床决策。""",
     """(3)~输入级错误（如源模态用错）超出采样器原生信号的感知范围，需要独立的输入防御层
——我们已将其开源实现于代码库，但超出本文论述范围。
(4)~图像级 QC 与肿瘤定位在此规模仍未解决；信任图还不响应输入退化——模糊反而抬高它
（\\S\\ref{sec:exp-stress}）——退化输入检测由输入侧 Mahalanobis 检测承担，小幅错位两者皆盲。
(5)~全部输出为研究演示，不得指导临床决策。"""),
    # 10) 结论：像素级免训练定位
    ("""我们把最先进扩散桥采样器中被丢弃的内部量转化为像素级信任图、信任分与 QC 工作流，""",
     """我们把最先进扩散桥采样器中被丢弃的内部量构建了像素级的免训练可信度量化层，"""),
    # 11) 表1 caption 补 N
    ("""IXI：自训 128$\\times$128。盆腔：自训，各 80 测试切片。""",
     """IXI：自训 128$\\times$128（各 60 测试切片）。盆腔：自训，各 80 测试切片。"""),
    # 12) 表2 caption 补性质标注
    ("""Sep 只作数量级陈述。\\textbf{下：}官方权重上同一协议，信号排序改变且 T2$\\to$T1 全盲。}""",
     """Sep 只作数量级陈述。\\textbf{下：}官方权重上同一协议，信号排序改变且 T2$\\to$T1 全盲。
各行性质：随机=对照；梯度/输入噪声/TTA=等预算基线；CReF/仅集成/融合=本文方法。}"""),
]

# ================= 英文稿 =================
en_pairs = [
    # 1) Abstract
    ("""Fused into a
pixel-wise trust map with a three-tier QC gate and a six-class input gate
with out-of-distribution rejection, they deliver, under the honest protocol:""",
     """Fused into a pixel-wise trust map, they deliver, under the within-body protocol:"""),
    # 2) Contribution 1
    ("""fused into a pixel-wise \\TrustMap{} with a three-tier QC
        gate, plus a six-class input gate with Mahalanobis
        out-of-distribution rejection that hard-blocks mismatched or anomalous
        inputs before synthesis (\\S\\ref{sec:method}).""",
     """fused into a pixel-wise \\TrustMap{} (\\S\\ref{sec:method})."""),
    # 3) Contribution 3: collapse blindness
    ("""official-weights T2$\\to$T1 signal failure, and the semantic-proxy
        null result---and use them to delineate the method's""",
     """official-weights T2$\\to$T1 signal failure, the semantic-proxy
        null result, and the trust layer's blindness to a collapsed base
        model---and use them to delineate the method's"""),
    # 4) §3.4 title + body
    ("""\\subsection{Trust map, Trust Score, and QC gate}""",
     """\\subsection{Trust map fusion}"""),
    ("""The image-level $\\TrustScore$ is the body-region mean
of $\\TrustMap$ scaled to 0--100, gated by a three-tier rule (PASS/REVIEW/
REJECT); \\S\\ref{sec:exp} analyzes the operating points actually achieved by
these thresholds on three cohorts. A per-slice report (PNG + JSON) records the
maps, the per-step recursion statistics, and the input-gate verdict.""",
     """The feasibility of image-level aggregation (the body-region mean) is
analyzed as a negative result in \\S\\ref{sec:exp} (image-level QC operating
points); per-slice archiving (trust maps and per-step recursion statistics)
is provided with the open-source repository."""),
    # 5) §3.5 Input gate v2 paragraph delete
    ("""\\paragraph{Input gate v2.}
Before any synthesis, a six-class classifier over body-standardized percentile
and gradient features (brain T1/T2/FLAIR, pelvis T1w/T2w, pelvis CT) verifies
that the input matches the declared source modality, with a Mahalanobis
out-of-distribution test (threshold at the validation 99.5th percentile)
and a flat-image hard rule. In a 1{,}160-case validation matrix spanning
matched, cross-modality, cross-anatomy, and synthetic-OOD suites, \\emph{no}
illegal input (0/562) received a passing verdict, while no valid input was
false-rejected. Mismatched inputs to brain tasks are hard-blocked; mismatched
inputs to pelvis-CT tasks are warned and annotated, because unseen-patient
pelvic T1w slices are confused with brain classes in $52.5\\%$ of cases
(\\S\\ref{sec:discussion}).

""", ""),
    # 7) §4.9 stress: gate as instrument
    ("""the \\emph{input gate's}
Mahalanobis distance fires dose-dependently for blur ($2.9\\to15.0$) and noise
($\\to9.3$) but is blind to small misalignment ($\\to2.9$);""",
     """the input-side Mahalanobis distance (the open-sourced input-side defense
layer's OOD score, used here purely as a measurement instrument) fires
dose-dependently for blur ($2.9\\to15.0$) and noise ($\\to9.3$) but is blind
to small misalignment ($\\to2.9$);"""),
    ("""Knowing-what-you-don't-know here is carried by the \\emph{input gate}, not the
trust map; misalignment detection remains open (we discuss this in
\\S\\ref{sec:discussion}).""",
     """Knowing-what-you-don't-know here is carried by the input-side Mahalanobis
detector, not the trust map; misalignment detection remains open (we discuss
this in \\S\\ref{sec:discussion})."""),
    # 8) §4.8 tumor: principled blind spot
    ("""Flagged regions indicate where the model is
unsure about its synthesis, not where pathology is. The lesson goes""",
     """Flagged regions indicate where the model is
unsure about its synthesis, not where pathology is. This boundary is not an
implementation defect but a principled blind spot of variance-type signals.
The lesson goes"""),
    # 9) §5 opening: conditions of validity
    ("""\\section{Discussion}
\\label{sec:discussion}
\\paragraph{Reproducibility.}""",
     """\\section{Discussion}
\\label{sec:discussion}
\\paragraph{Conditions of validity.}
The trust layer works when: residual error is spatially structured, the base
model has not collapsed, and evaluation is performed within the body. It fails
under: texture-dominated error, bias-type hallucination (lesions), model-level
collapse (\\S\\ref{sec:exp-repro}), and image-level aggregation. We carry this
checklist through the discussion below.
\\paragraph{Reproducibility.}"""),
    # 10) Limitations rewrite
    ("""(3)~The input gate hard-blocks brain-task mismatches but only warns on pelvis
CT tasks: unseen-patient pelvic T1w is confused with brain classes in
$52.5\\%$ of cases, so hard blocking would reject half of the valid inputs;
the Mahalanobis OOD threshold is additionally inflated by cross-scanner
domain shift, leaving some structured synthetic probes to the classifier
layer rather than the distance test.
(4)~Image-level QC and tumor localization remain unsolved at this scale; the
trust map additionally does not respond to input degradation---blur even
raises it (\\S\\ref{sec:exp-stress})---so degraded-input detection relies on
the input gate, and small misalignments evade both.
(5)~The three-tier gate thresholds require per-deployment re-tuning
(\\S\\ref{sec:exp}).
(6)~All outputs are research demonstrations and must not guide clinical
decisions.""",
     """(3)~Input-level errors (a wrong source modality) are outside the perception
range of sampler-native signals; they require a dedicated input-side defense
layer, which we release in the open-source repository but exclude from this
paper's scope.
(4)~Image-level QC and tumor localization remain unsolved at this scale; the
trust map additionally does not respond to input degradation---blur even
raises it (\\S\\ref{sec:exp-stress})---so degraded-input detection relies on
the input-side Mahalanobis detector, and small misalignments evade both.
(5)~All outputs are research demonstrations and must not guide clinical
decisions."""),
    # 11) Conclusion: pixel-level training-free framing
    ("""We turned the discarded internals of a state-of-the-art diffusion-bridge
sampler into a pixel-wise trust map, score, and QC workflow for medical image
translation,""",
     """We built a pixel-wise, training-free trust-quantification layer from the
discarded internals of a state-of-the-art diffusion-bridge sampler,"""),
    # 12) Table 1 caption N
    ("""IXI:
self-trained 128$\\times$128. Pelvis: self-trained, 80 test slices each.""",
     """IXI:
self-trained 128$\\times$128 (60 test slices each). Pelvis: self-trained,
80 test slices each."""),
    # 13) Table 2 caption row roles
    ("""where the signal ranking changes and T2$\\to$T1 fails entirely.}""",
     """where the signal ranking changes and T2$\\to$T1 fails entirely.
Row roles: random=control; gradient/input-noise/TTA=
equal-budget baselines; CReF/ensemble/fused=ours.}"""),
]

patch(EN, en_pairs, "EN")
print("patch 1 complete")
