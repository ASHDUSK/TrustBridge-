# -*- coding: utf-8 -*-
"""v0.8.4 送审前修订：落实 GPT 审查 12 项（0 新增实验）。

必须项：§5 旧结论残留 / 摘要-§4.2 REJECT 矛盾 / 表1 成本拆分 / "正交"表述 /
LaTeX 残留根因（tex2html）/ 图4-表4 Sep 聚合差异说明
重要项：± 统计含义 / SynthRAD 患者划分 / CReF 归一化与固定分母
建议项：摘要引言结论软化 / 表2 配置范围 / 参考文献渲染
"""
import io
import re

L = r"D:\MedTranslateProject\docs\latex"


def patch(path, pairs, tag):
    s = io.open(path, encoding="utf-8").read()
    n_ok = 0
    for i, (old, new) in enumerate(pairs):
        if s.count(old) == 1:
            s = s.replace(old, new, 1)
            n_ok += 1
        else:
            print(f"[{tag}] #{i} count={s.count(old)}: {old[:60]!r}")
    io.open(path, "w", encoding="utf-8").write(s)
    print(f"{tag}: {n_ok}/{len(pairs)} applied")


# ================= related_work_zh.tex =================
rwz = [
 (r"""无需注入额外随机性；(ii) 我们配对了一个与采样方差正交的免费信号——CReF，""",
  r"""无需注入额外随机性；(ii) 我们配对了一个不同于跨链采样方差、直接来源于递归收敛过程的免费信号——CReF，"""),
 (r"""TrustBridge（本文） & 否 & 否 & \textbf{无额外}（单链） & 像素级 \\""",
  r"""TrustBridge-CReF（单链） & 否 & 否 & \textbf{无额外}（10 次前传） & 像素级 \\
TrustBridge-Fusion（$N{=}4$） & 否 & 否 & 40 次前传 & 像素级 \\"""),
 (r"""TrustBridge 一行的"无额外前传"限定于当前 SelfRDB 实现——其递归结构为常规 DDPM 所无。}""",
  r"""TrustBridge 的"无额外前传"仅指 CReF 单链信号（10 次前传）且限定于当前 SelfRDB 实现——其递归结构为常规 DDPM 所无；完整融合（CReF + $N{=}4$ 集成）需 40 次生成器前传，与各多次前向方法同量级。}"""),
]

# ================= related_work.tex =================
rwe = [
 (r"""TrustBridge is specific to the current SelfRDB implementation --- its
recursion structure does not exist in vanilla DDPMs.}""",
  r"""The "no extra" entry refers to the single-chain CReF signal (10 generator
calls) and is specific to the current SelfRDB implementation --- its recursion
structure does not exist in vanilla DDPMs; the full fusion (CReF + $N{=}4$
ensemble) costs 40 generator calls, the same order as the multi-forward
methods.}"""),
 (r"""TrustBridge (ours) & no & no & \textbf{none} (single chain) & pixel-level \\""",
  r"""TrustBridge-CReF (single chain) & no & no & \textbf{none} (10 calls) & pixel-level \\
TrustBridge-Fusion ($N{=}4$) & no & no & 40 calls & pixel-level \\"""),
]

# ================= main_zh.tex =================
zh = [
 # 1 摘要 REJECT 矛盾
 (r"""但只到 REVIEW、从不 REJECT，像素级 AUROC 降至 $0.61$（健康 $0.82$）；最初报告的""",
  r"""整体均值仍处于 REVIEW 区间（20 个测试切片中 18 个 REVIEW、2 个 REJECT），像素级 AUROC 降至 $0.61$（健康 $0.82$）；最初报告的"""),
 # 2 §4.2 措辞
 (r"""$0.61\pm0.03$（健康 $0.82$）——信号方向正确但远不够：结构错误的模型仍落在
REVIEW（0 PASS / 18 REVIEW / 2 REJECT），从不 REJECT，最初报告的全体 PASS（$94/100$，整图均值）同样被可信空气虚高。""",
  r"""$0.61\pm0.03$（健康 $0.82$）——信号方向正确但远不够：结构错误的模型其均值分仍落在
REVIEW 区间（逐片为 0 PASS / 18 REVIEW / 2 REJECT），无法据此可靠拒绝该模型；最初报告的全体 PASS（$94/100$，整图均值）同样被可信空气虚高。"""),
 # 3 §5 局限(1) 旧结论残留
 (r"""(1)~\TrustMap{}标记\emph{不可靠}区域而非\emph{错误}区域：自信合成的幻觉对它不可见——
且这种盲区延伸到模型级：自信地犯错的崩塌基座模型获得 PASS（平均信任 $94/100$，
\S\ref{sec:exp-repro}）。""",
  r"""(1)~\TrustMap{}标记\emph{不可靠}区域而非\emph{错误}区域：对于模型高度自洽但内容错误的
偏差型幻觉，采样器原生信号仍可能失效。这一盲区延伸到模型级：崩塌官方模型的体区平均
信任分仅为 $62.4/100$（健康自训模型 $82.4/100$），20 个测试切片中 18 个进入 REVIEW、
2 个进入 REJECT——信任层能对部分模型失效产生响应，但尚不能可靠完成模型级故障识别
（\S\ref{sec:exp-repro}）。"""),
 # 4 成本表引用
 (r"""前传（递归本来就要算；缓存与归一化的开销见成本表）。两类信号与采样链的关系见""",
  r"""前传（递归本来就要算），但仍存在残差缓存、存储与归一化带来的额外实现开销。两类信号与采样链的关系见"""),
 # 5 表2 caption 范围
 (r"""\caption{实现与分析配置（全部实验统一；推理种子 2026）。}""",
  r"""\caption{实现与分析配置（主实验默认配置；推理种子 2026。跨模型分辨率、压力测试与 $K_0$ 敏感性等偏离默认之处在相应小节单独说明。）}"""),
 # 6 引言瓶颈句软化
 (r"""（而非图像质量）才是部署的主要障碍。""",
  r"""（而非图像质量）可能是部署的主要障碍之一。"""),
 # 7 局部阅读
 (r"""输出支持放射科医生的局部阅读。""",
  r"""输出可为放射科医生的局部人工复核提供候选空间证据。"""),
 # 8 结论引擎句软化
 (r"""收割这些证据能把合成引擎变成
可评估信任的系统；""",
  r"""收割这些证据能为合成输出提供空间化的风险排序证据；"""),
 # 9 摘要陷阱句软化
 (r"""两个隐蔽的陷阱正在腐蚀医学图像翻译领域大多数不确定性量化（UQ）主张。""",
  r"""我们识别出两个可能系统性地扭曲医学图像翻译不确定性量化（UQ）结论的问题。"""),
 # 10 ± 统计含义（§4.1 定义段尾）
 (r"""AUSE 为模型与 oracle 风险-覆盖曲线之差的逐片宏平均积分。""",
  r"""AUSE 为模型与 oracle 风险-覆盖曲线之差的逐片宏平均积分。全文 $\pm$ 均为逐片指标在评价切片上的标准差（切片级）；患者级不确定性以 $B{=}5000$、患者为重采样单位的配对 Bootstrap CI 单独报告（\S\ref{sec:exp-uq}）。BraTS 6 例、SynthRAD 2/1 例——切片数不可解读为独立生物学样本数。"""),
 # 11 SynthRAD 患者划分
 (r"""运行完全相同的冻结管线（T1w$\to$CT：160 切片、2 患者；T2w$\to$CT：80 切片、
1 患者；使用全部已制备 SynthRAD 切片、含指定验证划分，不重调任何配置）。""",
  r"""运行完全相同的冻结管线（T1w$\to$CT：160 切片、2 患者；T2w$\to$CT：80 切片、
1 患者；使用全部已制备 SynthRAD 切片、含指定验证划分，不重调任何配置）。
评价患者与训练患者完全不重叠（T1w$\to$CT 训练 5 例患者，评价 1PA148/1PA173；
T2w$\to$CT 评价 1PC069）；本评价零调参，验证划分不构成选择泄漏（训练侧 C 中心
验证划分与训练共享患者，属训练管线事项，已另行文档化）。"""),
 # 12 CReF 归一化与固定分母（§3.2）
 (r"""分母为固定 $K_0 N$——步长跳跃使实际捕获步数可少于 $K_0$，此差异如实保留而非调整分母
（公式与代码逐像素一致，4$\times$4 输入手算核对）。""",
  r"""分母为固定 $K_0 N$——步长跳跃使实际捕获步数可少于 $K_0$，此差异如实保留而非调整分母
（公式与代码逐像素一致，4$\times$4 输入手算核对）。归一化的分位数在每片整图（含背景）
上计算，CReF 图与 $U$ 图各一次；捕获不足的步不贡献残差，故捕获配置不同的图之间
CReF 绝对值不可直接比较。"""),
 # 13 fig:decile caption 聚合差异
 (r"""最高三箱误差接近零，Spearman 在零值段不敏感。}""",
  r"""最高三箱误差接近零，Spearman 在零值段不敏感。图内首尾箱标注（196.9$\times$/3.8$\times$）
为各箱跨片中位误差之比，与表~4 的逐片首尾箱比中位数（182$\times$/3.7$\times$）聚合
方式不同，故数值略有差异。}"""),
 # 14 §4.6 结论范围
 (r"""\emph{全局标量信任分根本不应作为医学图像翻译的
QC 门}。""",
  r"""\emph{全局标量信任分不应作为医学图像翻译的
QC 门}（本结论基于当前三个队列与所示操作点）。"""),
]

# ================= main.tex =================
en = [
 (r"""the healthy model's $82$, yet only REVIEW, never REJECT, and its pixel-level""",
  r"""the healthy model's $82$---its mean score sits in the REVIEW band (per-slice:
0 PASS / 18 REVIEW / 2 REJECT)---and its pixel-level"""),
 (r"""lands in REVIEW (0 PASS / 18 REVIEW / 2 REJECT), never REJECT, and the originally reported blanket PASS
($94/100$, a whole-image mean) was itself inflated by trusted air.""",
  r"""lands in the REVIEW band (per-slice: 0 PASS / 18 REVIEW / 2 REJECT) and cannot
reliably be rejected on this basis; the originally reported blanket PASS
($94/100$, a whole-image mean) was itself inflated by trusted air."""),
 (r"""signal requires no extra generator forwards (the recursion runs anyway; caching
and normalization overheads are tabulated in the cost table).""",
  r"""signal requires no extra generator forwards (the recursion runs anyway), though
residual caching, storage and normalization remain implementation overheads."""),
 (r"""\caption{Implementation and analysis configuration (one configuration for
all experiments; inference seed 2026 throughout).}""",
  r"""\caption{Implementation and analysis configuration (default settings for the
main experiments; inference seed 2026 throughout. Deviations --- cross-model
resolutions, stress tests, $K_0$ sensitivity --- are stated in their sections.)}"""),
 (r"""this trust bottleneck---not image quality---is the primary obstacle to
deployment.""",
  r"""this trust bottleneck---not image quality---may be a primary obstacle to
deployment."""),
 (r"""hallucination metrics~\cite{tivnan2024hallucinationindex}, the output supports
a radiologist's local reading.""",
  r"""hallucination metrics~\cite{tivnan2024hallucinationindex}, the output is
intended to provide candidate-region evidence for local human review."""),
 (r"""harvesting it converts a
synthesis engine into a trust-assessable system; but uncertainty claims in""",
  r"""harvesting it provides spatial
risk-ranking evidence for the synthesis output; but uncertainty claims in"""),
 (r"""Two pitfalls quietly corrupt most uncertainty-quantification (UQ) claims made
for generative medical image translation.""",
  r"""Two silent pitfalls can systematically bias most uncertainty-quantification
(UQ) claims made for generative medical image translation."""),
 (r"""AUSE is
the per-slice macro-averaged integral of the model-minus-oracle risk--coverage
difference.""",
  r"""AUSE is
the per-slice macro-averaged integral of the model-minus-oracle risk--coverage
difference. Throughout, $\pm$ denotes the slice-level standard deviation across
evaluated slices; patient-level uncertainty is reported separately as
paired-bootstrap CIs ($B{=}5000$, patients as resampling units,
\S\ref{sec:exp-uq}). Cohort sizes: BraTS 6 patients, SynthRAD 2/1 patients ---
slice counts must not be read as independent biological samples."""),
 (r"""T2w$\to$CT: 80 slices, 1 patient; all prepared SynthRAD slices, including
the designated validation split, with no configuration re-tuned).""",
  r"""T2w$\to$CT: 80 slices, 1 patient; all prepared SynthRAD slices, including
the designated validation split, with no configuration re-tuned). Evaluated
patients are fully disjoint from training patients (T1w$\to$CT: 5 training
patients vs.\ 1PA148/1PA173; T2w$\to$CT: 1PC069); no tuning occurred, so the
validation split does not constitute selection leakage (the training-side
center-C val split shares patients with training --- a training-pipeline
matter, documented separately)."""),
 (r"""into the denominator (the formula matches the code pixel-for-pixel, verified by
a hand calculation on a $4\times4$ input).""",
  r"""into the denominator (the formula matches the code pixel-for-pixel, verified by
a hand calculation on a $4\times4$ input). The percentile statistics are
computed per slice over the whole map (background included), once for the CReF
map and once for $U$; uncaptured steps contribute no residual, so CReF values
are not comparable across maps with different capture configurations."""),
 (r"""near-zero error, where Spearman is insensitive.}""",
  r"""near-zero error, where Spearman is insensitive. The in-figure annotations
(196.9$\times$/3.8$\times$) are ratios of cross-slice median per-bin errors,
whereas Table~\ref{tab:uq} reports the median of per-slice bin-ratios
(182$\times$/3.7$\times$) --- different aggregations of the same data.}"""),
 (r"""re-tuning'': \emph{global scalar trust scores should not be used as QC
gates for medical image translation at all}.""",
  r"""re-tuning'': \emph{global scalar trust scores should not be used as QC
gates for medical image translation} (on the current cohorts and operating
points)."""),
]

# ================= tex2html.py：emph 跨数学区间 + 参考文献卷页渲染 =================
def patch_tex2html():
    p = r"D:\MedTranslateProject\scripts\tex2html.py"
    s = io.open(p, encoding="utf-8").read()

    # 整函数替换 _cmds：占位符方案（支持 \emph{...$x$...} 跨数学区间）
    m = re.search(r"def _cmds\(c, s\):.*?\n    return \"\"\.join\(parts\)\n", s, re.S)
    assert m, "_cmds not found"
    new_cmds = '''def _cmds(c, s):
    # 数学保护（占位符版）：先把 $...$ 区间换为占位符，使命令替换可跨数学区间
    # 匹配（修复 \\emph{...$x$...} 被切断的渲染 bug），最后原样还原数学区间。
    math_spans = []
    def _stash(m):
        math_spans.append(m.group(0))
        return "\\x00M%d\\x00" % (len(math_spans) - 1)
    s = re.sub(r"\\$[^$]*\\$", _stash, s)
    s = re.sub(r"\\\\emph\\{([^{}]*)\\}", r"<em>\\1</em>", s)
    s = re.sub(r"\\\\textbf\\{([^{}]*)\\}", r"<b>\\1</b>", s)
    s = re.sub(r"\\\\texttt\\{([^{}]*)\\}", r"<code>\\1</code>", s)
    s = re.sub(r"\\\\url\\{([^{}]*)\\}", r'<a href="\\1">\\1</a>', s)
    s = s.replace("\\\\TrustMap", "信任图" if getattr(c, "zh", False) else "trust map")
    s = s.replace("\\\\TrustScore", "Trust Score")
    s = s.replace("\\\\cref", "Convergence Residual Field (CReF)")
    s = s.replace("---", "—").replace("--", "–")
    s = s.replace("``", "“").replace("''", "”")
    s = s.replace("\\\\%", "%").replace("\\\\ ", " ").replace("vs\\\\.", "vs.")
    s = s.replace("\\\\S", "§").replace("\\\\,", " ").replace("~", " ")
    s = s.replace("\\\\\\\\", "<br>")
    s = re.sub(r"\\\\item\\s*", "", s)
    s = re.sub(r"\\{\\}?", "", s)
    s = re.sub(r"\\\\(emph|textbf|texttt|url|large|small|centering)\\b", "", s)
    s = re.sub(r"\\x00M(\\d+)\\x00",
               lambda m: math_spans[int(m.group(1))], s)
    return s
'''
    s = s[:m.start()] + new_cmds + s[m.end():]

    # ref_html 渲染卷/页/DOI
    old_ref = '''        seg = f"{a}{etal} ({year}). {title}{dot}"
        if venue:
            seg += f" <i>{venue}</i>."
        out.append(f"<li>{seg}</li>")'''
    new_ref = '''        seg = f"{a}{etal} ({year}). {title}{dot}"
        if venue:
            seg += f" <i>{venue}</i>."
        vol = _bib_field(body, "volume")
        pgs = _bib_field(body, "pages")
        doi = _bib_field(body, "doi")
        if vol:
            seg += f" {vol}"
        if pgs:
            seg += f": {pgs}"
        if doi:
            seg += f". doi:{doi}"
        out.append(f"<li>{seg}</li>")'''
    assert old_ref in s, "ref_html"
    s = s.replace(old_ref, new_ref, 1)
    io.open(p, "w", encoding="utf-8").write(s)
    print("tex2html: _cmds restructured + ref_html upgraded")



if __name__ == "__main__":
    patch(L + r"\related_work_zh.tex", rwz, "RW-ZH")
    patch(L + r"\related_work.tex", rwe, "RW-EN")
    patch(L + r"\main_zh.tex", zh, "ZH")
    patch(L + r"\main.tex", en, "EN")
    patch_tex2html()
