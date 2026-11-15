# -*- coding: utf-8 -*-
"""v0.8.5 文风重构补丁：清除项目报告/宣传痕迹（依据 B 规范 v1.0 与 C 清单）。
约束：不改数字结论立场（摘要数字仅删冗余，不改保留值）。"""
import io
import re

L = r"D:\MedTranslateProject\docs\latex"


def flex(literal):
    """把字面量转成容忍换行/多空白的 regex（词间 \s+，词内转义）。"""
    words = literal.split()
    return r"\s+".join(re.escape(w) for w in words)


def sub_file(path, pairs, tag, regex_pairs=()):
    raw = io.open(path, encoding="utf-8", newline="").read()
    s = raw.replace("\r\n", "\n")  # 读时归一化 CRLF，写回统一 LF（与仓库一致）
    n = 0
    for old, new in pairs:
        c = s.count(old)
        if c == 1:
            s = s.replace(old, new, 1); n += 1
        else:
            print(f"[{tag}] literal count={c}: {old[:50]!r}")
    for lit, new, tag2 in regex_pairs:
        s2, k = re.subn(flex(lit), lambda m, _new=new: _new, s, count=1)
        if k == 1:
            s = s2; n += 1
        else:
            print(f"[{tag}] regex {tag2} matches={k}")
    io.open(path, "w", encoding="utf-8", newline="").write(s)
    print(f"{tag}: {n} edits")


# ---------------- EN ----------------
en_pairs = [
 (r"\large (system: TrustBridge --- a training-free trust layer with",
  r"\large (TrustBridge: a training-free trust layer with"),
 (r"""T2$\to$T1 weights (fused $0.57$; the TTA baseline retains weak signal,
$0.63$), where residual error is dominated by fine texture---which signal, if""",
  r"""T2$\to$T1 weights (the equal-budget TTA baseline retains weak signal), where
residual error is dominated by fine texture---which signal, if"""),
 (r"""AUROC $0.82$/$0.76$ (T1$\to$T2/T2$\to$T1, self-trained; $0.79$/$0.70$ for
a single chain at no extra cost), $0.62$ on real tumor patients, and $0.83$
\emph{cross-institution} on never-seen subjects.""",
  r"""AUROC $0.82$/$0.76$ (self-trained; a single chain reaches most of this at
no extra cost), $0.62$ on real tumor patients, and $0.83$ \emph{cross-institution}
on never-seen subjects."""),
 (r"""0 PASS / 18 REVIEW / 2 REJECT)---and its pixel-level
AUROC degrades to $0.61$ (healthy $0.82$); the originally reported blanket PASS""",
  r"""0 PASS / 18 REVIEW / 2 REJECT); the originally reported blanket PASS"""),
 (r"""sampler (SelfRDB) we harvest two training-free signals: the""",
  r"""sampler (SelfRDB) we extract two training-free signals: the"""),
 (r"""multi-chain divergence (signal 2) are harvested from the recursion and fused""",
  r"""multi-chain divergence (signal 2) are extracted from the recursion and fused"""),
 (r"""harvesting them adds no model, no training and no labels.}""",
  r"""extracting them adds no model, no training and no labels.}"""),
 (r"""and harvesting it provides spatial""",
  r"""and deriving it provides spatial"""),
 (r"""as a
boundary map for the field, and recommend UQ claims in image translation be""",
  r"""as negative results delimiting the applicability of such methods, and
recommend UQ claims in image translation be"""),
 (r"""reproduction checklist (docs/18, with verified/unverified status and
evidence).""",
  r"""documented reproduction checklist with verified/unverified status and
evidence."""),
 (r"""and we publish our retraction trail (docs/18 and the corrected analysis
scripts) as evidence that this is easy to get wrong---we did.""",
  r"""and we provide our retraction trail and the corrected analysis scripts as
evidence that this is easy to get wrong---we did."""),
 (r"""PASS/REVIEW gates at 75/55 (demo defaults, uncalibrated)""",
  r"""PASS/REVIEW gates at 75/55 (default values, not clinically calibrated)"""),
 (r"""The "no extra" entry of
The "no extra" entry refers to the single-chain CReF signal""",
  r"""The "no extra" entry refers to the single-chain CReF signal"""),
]
en_regex = [
 (r"""The lesson for deployment is starker for being graded: \emph{training-free internal signals cannot be trusted to reject a base model that is confidently wrong}---a 20-point trust gap is a signal, not a gate.""",
  r"""The deployment implication is correspondingly limited: \emph{training-free internal signals should not be relied upon to reject a base model that is confidently wrong}.""",
  "slogan42"),
 (r"""The lesson goes beyond our method: pixel-variance-style UQ measures the \emph{variance of the synthesis process}, not the \emph{clinical salience} of the content---a model can be highly confident (and highly consistent) about a pathologically wrong synthesis. Any UQ layer for clinical use must be audited against pathology annotations, not only against error maps (Figure~\ref{fig:tumor}).""",
  r"""This boundary generalizes: pixel-variance-style UQ measures the variance of the synthesis process rather than the clinical salience of its content, which is why clinical UQ layers require audit against pathology annotations (\S\ref{sec:discussion}, Figure~\ref{fig:tumor}).""",
  "tumor-lesson"),
]

# ---------------- ZH ----------------
zh_pairs = [
 (r"\large （系统：TrustBridge——含收敛残差场的免训练信任层）}",
  r"\large （TrustBridge：含收敛残差场的免训练信任层）}"),
 (r"""官方 T2$\to$T1 权重上跌至随机水平（融合 $0.57$；TTA 基线保留弱信号 $0.63$）。那么，什么\emph{能够}幸存？""",
  r"""官方 T2$\to$T1 权重上跌至随机水平（等预算 TTA 基线保留弱信号）。那么，什么\emph{能够}幸存？"""),
 (r"""AUROC $0.82$（T1$\to$T2 自训）与 $0.76$
（T2$\to$T1 自训），超过全部等预算基线，单链零额外成本即达 $0.79$/$0.70$；""",
  r"""AUROC $0.82$（T1$\to$T2 自训）与 $0.76$（T2$\to$T1 自训），超过全部等预算基线，
单链以零额外成本达到其大部分；"""),
 (r"""像素级 AUROC 降至 $0.61$（健康 $0.82$）；最初报告的
全体 PASS（$94/100$，整图均值）本身就是空气池化伪影。""",
  r"""像素级 AUROC 亦同步退化；最初报告的
全体 PASS（$94/100$，整图均值）本身就是空气池化伪影。"""),
 (r"""（SelfRDB）中我们收割两类免训练信号：""",
  r"""（SelfRDB）中我们提取两类免训练信号："""),
 (r"""算，收割它们不引入任何模型、训练或标注。}""",
  r"""算，提取它们不引入任何模型、训练或标注。}"""),
 (r"""作为给领域的边界地图，并建议图像翻译的 UQ 主张一律在体区内、""",
  r"""作为界定此类方法适用范围的阴性结果，并建议图像翻译的 UQ 主张一律在体区内、"""),
 (r"""（docs/18 与修正脚本）作为方法论证据发表。""",
  r"""与修正后的分析脚本一并随论文提供，作为方法论证据。"""),
 (r"""均已记录在复现排查表中（docs/18，区分已核查/未核查与证据）。伴随的 SynDiff""",
  r"""均已记录在一份复现排查清单中（区分已核查/未核查与证据）。伴随的 SynDiff"""),
 (r"""并公开撤回轨迹（docs/18 与修正脚本）作为"这很容易做错"的证据——我们就做错过。""",
  r"""并公开撤回轨迹与修正后的分析脚本，作为"这很容易做错"的证据——我们就做错过。"""),
 (r"""PASS/REVIEW 门限 75/55（演示默认，未标定）""",
  r"""PASS/REVIEW 门限 75/55（默认值，未经临床标定）"""),
]
zh_regex = [
 (r"""部署教训因有层次而更冷峻：\emph{免训练内部信号不足以否决"自信地犯错"的基座模型}——20 分的信任差是一个信号，不是一道门。""",
  r"""部署教训因有层次而更冷峻：\emph{免训练内部信号不足以否决"自信地犯错"的基座模型}。""",
  "slogan-zh"),
 (r"""教训超出本文方法：像素方差式 UQ 度量的是\emph{合成过程的方差}，不是内容的\emph{临床显著性}——模型可以对病理上错误的合成高度自信（且高度自洽）。任何面向临床的 UQ 层都必须对照病理标注审计，而不能只对照误差图（图~\ref{fig:tumor}）。""",
  r"""这一边界是方差型信号原理性盲区的推广：像素方差式 UQ 度量的是合成过程的方差而非内容的临床显著性，因此面向临床的 UQ 层必须对照病理标注审计（\S\ref{sec:discussion}，图~\ref{fig:tumor}）。""",
  "tumor-lesson-zh"),
]

rw_caption = [
 (r"""generator calls beyond a single base inference). The "no extra" entry of
The "no extra" entry refers to the single-chain CReF signal""",
  r"""generator calls beyond a single base inference). The "no extra" entry refers
to the single-chain CReF signal"""),
]

if __name__ == "__main__":
    sub_file(L + r"\main.tex", en_pairs, "EN", en_regex)
    sub_file(L + r"\main_zh.tex", zh_pairs, "ZH", zh_regex)
    sub_file(L + r"\related_work.tex", rw_caption, "RW-EN")
