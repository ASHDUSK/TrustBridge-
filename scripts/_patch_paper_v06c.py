# -*- coding: utf-8 -*-
"""论文 v0.6 patch 3：模拟审稿 P0/P1 修复（中英双稿）。"""
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


zh_pairs = [
    # P0-N1: 摘要 0.93 → 0.90（统一为受控对照的口径演示数字）
    ("""在体掩码口径下从 $0.93$ 崩落到 $0.78$""",
     """在体掩码口径下从 $0.90$ 崩落到 $0.78$"""),
    # P0-N2: 预算 50 → 删除（表2 口径为 40）
    ("""（预算 10，$0.880$）已全面优于满预算 TTA（预算 50，$0.820$）——路由在该模型上没有增益，""",
     """（预算 10，$0.880$）已全面优于满预算 TTA（$0.820$）——路由在该模型上没有增益，"""),
    # P0-N3: §5 +3-4dB 无出处 → 重写该段（以盆腔/官方T2→T1 实测为准）
    ("""\\paragraph{质量与信任校准可分离。}
BraTS 上自训模型 PSNR 高于官方权重 $+3$--$4$\\,dB，官方权重的信任图却定位得更准。
T2$\\to$T1 收紧了论点：模型可以准确而其不确定性无信息——当残差误差缺乏分歧度量所响应的
空间结构时。不确定性信号必须按（模型, 方向）对逐一验证后才可部署——这一结论由我们自己的
阴性结果落地。""",
     """\\paragraph{质量与信任校准可分离。}
自训盆腔桥 PSNR 高于官方权重 $+3.1$/$+4.1$\\,dB，但信号排序随模型改变（表2 上/下面板）；
最尖锐的例子是官方 T2$\\to$T1 权重——它的信任信号完全无信息。模型可以准确而其不确定性
无信息：当残差误差缺乏分歧度量所响应的空间结构时。不确定性信号必须按（模型, 方向）对
逐一验证后才可部署——这一结论由我们自己的阴性结果落地。"""),
    # P1-N7: §4.9 注明模型与口径
    ("""翻译质量轻微恶化（体区 PSNR 最差剂量 $12.9\\to11.8$\\,dB）。信任机制的响应不对称而有
教学意义：输入侧 Mahalanobis 距离""",
     """翻译质量轻微恶化（体区 PSNR 最差剂量 $12.9\\to11.8$\\,dB；官方 T1$\\to$T2 权重；
本节信任值为体区均值，聚合口径与 \\S4.2 的整图信任分不同）。信任机制的响应不对称而有
教学意义：输入侧 Mahalanobis 距离"""),
    # P1-B1: §2.3 §5 → §6（related_work_zh.tex，单独处理）
    # P1-B2: §4.1 撤回指向
    ("""均已重评，其中两项被撤回（\\S\\ref{sec:exp-tumor}）。""",
     """均已重评，其中两项被撤回（\\S\\ref{sec:exp-tumor}、\\S4.5）。"""),
    # P1-E4: ≥20 软化（§4.7）
    ("""像素级\\TrustMap{}是唯一可靠的产品形态，可学习聚合需要 $\\ge$20 验证被试。""",
     """像素级\\TrustMap{}是唯一可靠的产品形态；重开可学习聚合需要大得多的验证集（估计 $\\ge$20 被试，待验证）。"""),
    # P1-E4: §5 局限(2)
    ("""(2)~可学习聚合过拟合 6 被试验证集；重开需 $\\ge$20 验证被试。""",
     """(2)~可学习聚合过拟合 6 被试验证集；重开需要大得多的验证集（估计 $\\ge$20 被试，待验证）。"""),
    # P1-E4: §6 结论
    ("""未来工作将在更大被试规模上重开可学习聚合、""",
     """未来工作将在更大被试规模上检验是否可重开可学习聚合、"""),
    # P1-E5: 口径陷阱全称软化
    ("""凡图像有"可信低误差背景"的翻译设定，整图口径的逐像素信任指标都会系统性乐观——这几乎
覆盖全部医学图像翻译场景。""",
     """在图像存在"可信低误差背景"的翻译设定中——绝大多数医学图像翻译场景皆如此——
整图口径的逐像素信任指标会系统性乐观。"""),
    # P1-N8: zh §4.8 已用 90%（无需改）
    # 贡献 3 计数修正（八项实为 7+1 结构）
    ("""  \\item 我们报告七项结果为混合或阴性的受控实验——自适应重采样、可学习聚合、图像级回归、
        强度调和、往返一致信号、官方权重 T2$\\to$T1 信号失效、语义代理无互补、以及基座模型崩塌时信任层失明——
        并以此划定方法边界：分歧信号只在误差呈空间结构时有效，可学习聚合在
        6 被试验证集上过拟合（\\S4、\\S5）。""",
     """  \\item 我们报告七项边界实验（\\S4.7）与一项模型级失效——基座模型崩塌时信任层失明
        （\\S4.2）——结果为混合或阴性：自适应重采样、可学习聚合、图像级回归、强度调和、
        往返一致信号、官方权重 T2$\\to$T1 信号失效、语义代理无互补。
        并以此划定方法边界：分歧信号只在误差呈空间结构时有效，可学习聚合在
        6 被试验证集上过拟合（\\S4、\\S5）。"""),
    # §4.6 补 9%/29%（对齐英文）
    ("""REJECT$<55$ 仅覆盖 $3\\%$。""",
     """REJECT$<55$ 仅覆盖 $3\\%$；最优汇总工作点上也只能以 9% 覆盖换取 29% 精确率
（对质量下四分位）。"""),
]

en_pairs = [
    # P0-N1
    ("""score of $0.93$ collapsing to $0.78$ under a within-body protocol""",
     """score of $0.90$ collapsing to $0.78$ under a within-body protocol"""),
    # P0-N2
    ("""full TTA ($0.820$ at budget 50)---so routing adds nothing there, a further""",
     """full TTA ($0.820$)---so routing adds nothing there, a further"""),
    # P0-E1: EN abstract routing → negative + add T2→T1 numbers
    ("""AUROC $0.90$ (T1$\\to$T2, self-trained; $0.88$ for a single zero-cost chain),
$0.91$ on real tumor patients, and $0.95$ \\emph{cross-institution} on
never-seen subjects. CReF-triggered routing recovers most of full-budget TTA
quality at a fraction of its cost.""",
     """AUROC $0.90$/$0.82$ (T1$\\to$T2/T2$\\to$T1, self-trained; $0.88$/$0.79$ for
a single zero-cost chain), $0.91$ on real tumor patients, and $0.95$
\\emph{cross-institution} on never-seen subjects. A CReF-triggered routing
scheme adds nothing on these models---single-chain CReF already dominates
full-budget TTA---further confirming that which signal wins is
model-dependent."""),
    # P0-N3: §5 calibration paragraph rewrite
    ("""\\paragraph{Quality and trust calibration are separable.}
On BraTS, our self-trained models beat the official weights on PSNR by
$+3$--$4$\\,dB, yet the official weights' trust maps localize error far better.
The T2$\\to$T1 result sharpens the point: a model can be accurate and still
produce uninformative uncertainty, when its residual error lacks the spatial
structure that dispersion measures respond to. Uncertainty signals must be
validated per (model, direction) pair before deployment---a conclusion our
own negative result makes concrete.""",
     """\\paragraph{Quality and trust calibration are separable.}
Our self-trained pelvis bridges beat the official weights on PSNR by
$+3.1$/$+4.1$\\,dB, yet the signal ranking changes across models (top/bottom
panels of Table~\\ref{tab:uq}); the sharpest example is the official
T2$\\to$T1 weights, whose trust signals carry no information at all. A model
can be accurate and still produce uninformative uncertainty, when its
residual error lacks the spatial structure that dispersion measures respond
to. Uncertainty signals must be validated per (model, direction) pair before
deployment---a conclusion our own negative result makes concrete."""),
    # P1-E2: delete overbroad EN Reproducibility paragraph in §5 (redundant with §4.2)
    ("""\\paragraph{Reproducibility.}
Officially released IXI weights reproduce only $\\sim$15\\,dB on independently
prepared IXI data (the paper reports 31.56\\,dB). A systematic exclusion study
(rigid misalignment, $z$-shift, intensity window, cropping, orientation)
traces the gap to preprocessing details that the original authors do not
publish; the same issue is independently reported for the companion SynDiff
release. We therefore trained same-recipe models on our own fully documented
pipeline and base all quantitative claims on them, while using the official
weights for high-resolution demonstration.
\\paragraph{Quality and trust calibration are separable.}""",
     """\\paragraph{Quality and trust calibration are separable.}"""),
    # P1-E3: EN §2.2 calibrated/QC wording —— 在 related_work.tex，由 EN-RW 处理
    # P1-B1: EN §2.3 §5 → §6（related_work.tex，单独处理）
    # P1-B2: EN §4.1 retraction pointer
    ("""were re-evaluated; two were
retracted (\\S\\ref{sec:exp-tumor}).""",
     """were re-evaluated; two were retracted (\\S\\ref{sec:exp-tumor},
\\S\\ref{sec:exp-uq})."""),
    # P1-E4: EN ≥20 softening (§4.7)
    ("""at 40 subjects, the pixel-wise \\TrustMap{} is the only reliable product form,
and learning-based aggregation requires $\\ge$20 validation subjects.""",
     """at 40 subjects, the pixel-wise \\TrustMap{} is the only reliable product
form; reopening learnable aggregation would require a substantially larger
validation set (plausibly $\\ge$20 subjects, untested)."""),
    # P1-E4: EN contribution
    ("""learning-based aggregation overfits
        six-subject validation sets (\\S\\ref{sec:exp}, \\S\\ref{sec:discussion}).""",
     """learning-based aggregation overfits six-subject validation sets
        (\\S\\ref{sec:exp}, \\S\\ref{sec:discussion})."""),
    # P1-E4: EN limitations (2)
    ("""(2)~Learnable aggregation overfits 6-subject validation sets; reopening the
learnable route requires $\\ge$20 validation subjects.""",
     """(2)~Learnable aggregation overfits 6-subject validation sets; reopening it
would require a substantially larger validation set (plausibly $\\ge$20
subjects, untested)."""),
    # P1-E4: EN conclusion
    ("""Future work will reopen learnable
aggregation at larger subject scale, extend the analysis to 256$\\times$256
models, and integrate radiologist feedback.""",
     """Future work will test whether learnable aggregation can be reopened at
larger subject scale, extend the analysis to 256$\\times$256 models, and
integrate radiologist feedback."""),
    # P1-E5: EN protocol-trap universal claim
    ("""Whole-image evaluation of pixel-wise trust metrics is systematically optimistic
whenever images have trusted, low-error background. Because this describes
nearly all medical image translation settings,""",
     """Whole-image evaluation of pixel-wise trust metrics is systematically
optimistic whenever images have trusted, low-error background---as almost all
medical image translation settings do."""),
    # P1-N5: EN gradient Sep 0.40 → 0.51 (align with computed median)
    ("""Gradient & 0.225 & $-0.45$ & 0.54 & 0.475 & $-0.55$ & 0.40 \\\\""",
     """Gradient & 0.225 & $-0.45$ & 0.54 & 0.475 & $-0.55$ & 0.51 \\\\""",
     ),
    # P1-N7: EN §4.9 model+scale note
    ("""Translation quality degrades
mildly (body PSNR $12.9\\to11.8$\\,dB at the worst dose). The response of our""",
     """Translation quality degrades
mildly (body PSNR $12.9\\to11.8$\\,dB at the worst dose; official T1$\\to$T2
weights; trust values in this section are body means, a different aggregation
from the whole-image score of \\S\\ref{sec:exp-repro}). The response of our"""),
    # P1-N8: EN 90 of slices → 已是 90\%（无需改）
    # P1-N8: EN conclusion add T2→T1 to model list
    ("""on models whose errors are spatially structured (T1$\\to$T2; official
weights on tumor patients)""",
     """on models whose errors are spatially structured (self-trained
T1$\\to$T2/T2$\\to$T1; official weights on tumor patients)"""),
    # P1-N6: EN contribution 3 count structure
    ("""  \\item We report seven controlled experiments whose outcomes are mixed or
        negative---adaptive resampling, learnable aggregation, image-level
        regression, intensity harmonization, round-trip consistency, the
        official-weights T2$\\to$T1 signal failure, the semantic-proxy
        null result, and the trust layer's blindness to a collapsed base
        model---and use them to delineate the method's""",
     """  \\item We report seven controlled boundary experiments (\\S\\ref{sec:exp})
        plus a model-level failure---the trust layer's blindness to a collapsed
        base model (\\S\\ref{sec:exp-repro})---whose outcomes are mixed or
        negative: adaptive resampling, learnable aggregation, image-level
        regression, intensity harmonization, round-trip consistency, the
        official-weights T2$\\to$T1 signal failure, and the semantic-proxy
        null result. We use them to delineate the method's"""),
    # P1-N8: EN §4.2 NCC 0.9 → structural (align with 0.41 below)
    ("""and the failure
mode is invisible---the official outputs are structurally coherent images
with a global intensity offset (NCC ${\\sim}0.9$), not noise.""",
     """and the outputs are smooth, plausible-looking images rather than noise
---yet structurally wrong (within-body NCC $0.41$; see below)."""),
]

patch(EN, en_pairs, "EN")
patch(r"D:\MedTranslateProject\docs\latex\related_work_zh.tex",
      [("""它们与采样器原生证据互补，我们在 \\S5 作为未来工作讨论。""",
        """它们与采样器原生证据互补，我们在 \\S6 展望中讨论。""")],
      "ZH-RW")
patch(r"D:\MedTranslateProject\docs\latex\related_work.tex",
      [("""fuse both into a calibrated trust map with a clinical QC gate rather than
reporting dispersion alone.""",
        """fuse both into a pixel-wise trust map integrated with a QC workflow,
whose image-level semantics we then audit, rather than reporting dispersion
alone."""),
       ("""they are complementary to
sampler-native evidence and are discussed as future work in \\S\\ref{sec:discussion}.""",
        """they are complementary to
sampler-native evidence and are discussed as future work in \\S6.""")],
      "EN-RW")
print("patch 3 complete")
