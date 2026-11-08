# -*- coding: utf-8 -*-
"""论文 v0.7 patch C：全部新数字回填（摘要三点式 + 患者CI + K消融 + SynthRAD UQ + 健康退化对照 + 权重扫描）。"""
import io

ZH = r"D:\MedTranslateProject\docs\latex\main_zh.tex"
EN = r"D:\MedTranslateProject\docs\latex\main.tex"


def patch(path, pairs, tag):
    s = io.open(path, encoding="utf-8").read()
    for i, (old, new) in enumerate(pairs):
        assert old in s, f"[{tag}] #{i}: {old[:60]!r}"
        s = s.replace(old, new, 1)
    io.open(path, "w", encoding="utf-8").write(s)
    print(f"{tag}: {len(pairs)} edits")


def splice(path, start, end, new, tag="auto"):
    s = io.open(path, encoding="utf-8").read()
    if new[:60] in s:
        print(f"{tag}: skip"); return
    i0 = s.index(start); i1 = s.index(end, i0) + len(end)
    s = s[:i0] + new + s[i1:]
    io.open(path, "w", encoding="utf-8").write(s)
    print(f"{tag}: spliced")


# ================= ZH =================
zp = [
    # 摘要三点式重构
    ("""两个隐蔽的陷阱正在腐蚀医学图像翻译领域大多数不确定性量化（UQ）主张。其一，\\emph{背景空气像素}：
空气区域"高信任、近零误差"，混入整图统计后会成倍虚高 UQ 指标——我们展示同一信号的头条定位分数
在体掩码口径下从 $0.90$ 崩落到 $0.78$，并撤回了自己早期受益于此的数字。其二，\\emph{分歧不等于误差}：
在与 TTA、输入噪声基线的等预算对比中，所有分歧类信号——包括我们的——在残差误差由细纹理主导的
T2$\\to$T1 官方权重上全部等于随机水平。那么，什么\\emph{能够}幸存？从自洽递归扩散桥采样器
（SelfRDB）中我们收割两类免训练信号：\\emph{收敛残差场}（CReF，采样器本来就在计算却丢弃的
相邻递归目标估计逐像素差异）与\\emph{随机桥集成不确定性}。融合为逐像素\\TrustMap{}后，在体掩码口径下它们给出：AUROC $0.90$（T1$\\to$T2 自训）与 $0.82$
（T2$\\to$T1 自训），超过全部等预算基线，单链零额外成本即达 $0.88$/$0.79$；真实肿瘤患者
$0.91$；\\emph{跨机构}（从未见过的受试者）$0.95$。CReF 触发的路由在自训模型上无增益——
进一步确认信号排序的模型相关性。专门的复现性分析解释了官方权重为何在独立制备数据上崩塌至
${\\sim}15$\\,dB（论文值 31.56\\,dB）——是结构性的而不仅是光度的（NCC 0.41）——并表明信任层
\\emph{未能察觉}：崩塌模型获得 PASS 与 $94/100$ 的平均信任。自洽性无法证明正确性，
外部验证不可替代。我们发布全部阴性结果——口径陷阱、纹理主导失效边界、肿瘤定位缺位、
图像级门控失准——作为给领域的边界地图，并建议图像翻译的 UQ 主张一律在体区内、
对照廉价基线、按（模型, 方向）逐一评估。""",
     """方法：从自洽递归扩散桥采样器（SelfRDB）中收割两类免训练信号——\\emph{收敛残差场}
（CReF，采样器本来就在计算却丢弃的相邻递归估计逐像素差异）与\\emph{随机桥集成分歧}——
融合为逐像素信任图。主要结果（体掩码口径）：自训模型 T1$\\to$T2 AUROC $0.90$、T2$\\to$T1
$0.82$，超过全部等预算基线；真实肿瘤患者 $0.91$（95\\% CI $[0.89, 0.93]$）；跨机构
6 例初步验证 $0.95$。失效边界：融合相对仅集成的优势在 T1$\\to$T2 不显著（$\\Delta$
$+0.004$，95\\% CI $[-0.001, 0.008]$）；官方 T2$\\to$T1 权重上全部分歧类信号
\\emph{系统性低于随机}（CI 上界 $<0.5$）；信任不定位肿瘤（0.51）且对自信的基座模型崩塌
完全失明（PASS 94/100）。我们发布全部阴性结果作为给领域的边界地图，并建议 UQ 主张
一律在体区内、对照等预算基线、按（模型, 方向）逐一评估。""",),
    # §4.4 "全面获胜" → 按 CI 改写
    ("""\\textbf{自训模型上}，本文融合信号全面获胜：AUROC $0.900$/$0.816$（T1$\\to$T2/T2$\\to$T1）
        并以 $+2.67$/$+1.85$\\,dB 的稀疏化增益领先全部等预算基线，而单链 CReF
        （零额外成本）已达 $0.880$/$0.792$——以四分之一的预算获得大部分价值。融合
        稳健但并不比仅集成好多少；CReF 的主张是成本，而非精度。""",
     """\\textbf{自训模型上}，本文融合信号 AUROC $0.900$/$0.816$（T1$\\to$T2/T2$\\to$T1），
        配对患者级 bootstrap（$B{=}5000$，8 被试）显示：融合相对仅集成的 $\\Delta$AUROC
        为 $+0.004$（95\\% CI $[-0.001, 0.008]$，\\emph{含 0}——T1$\\to$T2 上不显著）与
        $+0.010$（CI $[0.005, 0.015]$，T2$\\to$T1 上显著）；而融合相对全部等预算基线
        （TTA $-0.082$/$-0.107$、噪声 $-0.041$/$-0.078$、梯度）差异显著且幅度大。
        CReF 单链（零额外成本）$0.880$/$0.792$——以四分之一预算获得大部分价值。""",),
    # §4.5 权重扫描 + K 消融
    ("""而非绝对误差量级（图~\\ref{fig:decile}）。""",
     """而非绝对误差量级（图~\\ref{fig:decile}）。捕获窗消融显示 K$\\in\\{1,\\ldots,10\\}$
        的 AUROC 变化 $<0.008$（$K{=}1$ 即 $0.811$，$K{=}10$ 为 $0.818$）——捕获窗长度
        不敏感。验证集权重扫描（$w_1\\in\\{0,\\ldots,1\\}$，6 被试）显示 $w_1{=}0.2$--$0.4$
        最优（$0.802$）、$w_1{=}0.6$ 为 $0.794$——默认 $0.6$ 并非验证集最优，但全曲线
        变化 $<0.05$，属稳健平台而非精调峰值。""",),
]

en_p = [
    ("""Fused into a pixel-wise trust map, they deliver, under the within-body protocol:
AUROC $0.90$/$0.82$ (T1$\\to$T2/T2$\\to$T1, self-trained; $0.88$/$0.79$ for
a single zero-cost chain), $0.91$ on real tumor patients, and $0.95$
\\emph{cross-institution} on never-seen subjects. A CReF-triggered routing
scheme adds nothing on these models---single-chain CReF already dominates
full-budget TTA---further confirming that which signal wins is
model-dependent.""",
     """Under the within-body protocol, the self-trained models reach AUROC
$0.90$/$0.82$ (T1$\\to$T2/T2$\\to$T1); real tumor patients $0.91$ (95\\% CI
$[0.89, 0.93]$); a preliminary cross-institution evaluation on six subjects
$0.95$. Failure boundaries: the fusion advantage over ensemble alone is
\\emph{not} significant on T1$\\to$T2 ($\\Delta$ $+0.004$, CI $[-0.001, 0.008]$);
on the official T2$\\to$T1 weights all dispersion signals are \\emph{systematically
below chance} (CI upper bound $<0.5$); trust does not localize tumors (0.51)
and is blind to confident base-model collapse (PASS 94/100). We release all
negative results as a boundary map for the field.""",),
    ("""\\textbf{On self-trained models},
our fused signal wins outright: AUROC $0.900$/$0.816$ (T1$\\to$T2/T2$\\to$T1)
with $+2.67$/$+1.85$\\,dB sparsification gains, ahead of every equal-budget
baseline, and a single CReF chain (zero extra cost) already reaches
$0.880$/$0.792$---most of the value at a quarter of the budget. Fusion is
robust but does not beat the ensemble alone by much; CReF's claim is cost,
not accuracy.""",
     """\\textbf{On self-trained models},
the fused signal achieves AUROC $0.900$/$0.816$ (T1$\\to$T2/T2$\\to$T1) with
$+2.67$/$+1.85$\\,dB sparsification gains. Paired subject-level bootstrap
($B{=}5000$, 8 subjects) shows: fused vs.\\ ensemble alone $\\Delta$AUROC
$+0.004$ (95\\% CI $[-0.001, 0.008]$, \\emph{includes zero} --- not significant
on T1$\\to$T2) and $+0.010$ (CI $[0.005, 0.015]$, significant on T2$\\to$T1);
fused vs.\\ all equal-budget baselines is significant with large margins
(TTA $-0.082$/$-0.107$, noise $-0.041$/$-0.078$, gradient). A single CReF
chain (zero extra cost) reaches $0.880$/$0.792$ --- most of the value at a
quarter of the budget. CReF's claim is cost, not accuracy over the ensemble.""",),
    ("""The trust map is best read as
a \\emph{ranking within the body}, not an absolute error magnitude (Figure~\\ref{fig:decile}).""",
     """The trust map is best read as a \\emph{ranking within the body}, not an
absolute error magnitude (Figure~\\ref{fig:decile}). Capture-window ablation:
AUROC varies $<0.008$ across $K\\in\\{1,\\ldots,10\\}$. Validation-set weight
sweep: $w_1{=}0.2$--$0.4$ optimal ($0.802$), $w_1{=}0.6$ gives $0.794$ ---
the default is on a robust plateau, not a tuned peak.""",),
]

patch(ZH, zp, "ZH-C")
patch(EN, en_p, "EN-C")
print("patch C complete")
