# -*- coding: utf-8 -*-
"""论文 v0.7 patch A2：用起止标记切片替换 §3.2/§3.3/§3.4（双语）。"""
import io

ZH = r"D:\MedTranslateProject\docs\latex\main_zh.tex"
EN = r"D:\MedTranslateProject\docs\latex\main.tex"


def splice(path, start_marker, end_marker, new_block, tag="auto"):
    s = io.open(path, encoding="utf-8").read()
    if new_block[:60] in s:
        print(f"{tag}: already applied, skip")
        return
    i0 = s.index(start_marker)
    i1 = s.index(end_marker, i0) + len(end_marker)
    s = s[:i0] + new_block + s[i1:]
    io.open(path, "w", encoding="utf-8").write(s)
    print(f"{tag}: spliced ({i1-i0} -> {len(new_block)} chars)")


# ================= ZH §3.2（起点=段首，终点=图引用句末）=================
splice(ZH,
       "每个反向步内，递归产生逐像素残差",
       "两类信号与采样链的关系见图~\\ref{fig:overview}。",
       """每个反向步内，递归产生逐像素残差；SelfRDB 仅取其全局均值作停机判据并丢弃空间信息。
我们保留捕获窗内各反向步的\\emph{末次递归残差}：
\\begin{equation}
  \\mathrm{CReF}(p) \\;=\\; \\frac{1}{K_0\\cdot N}\\sum_{i=1}^{N}\\sum_{t\\in\\mathcal T_i}
      \\bigl\\lvert \\tilde x_0^{(r_t,i)}(p)-\\tilde x_0^{(r_t-1,i)}(p)\\bigr\\rvert ,
  \\label{eq:cref}
\\end{equation}
其中 $\\mathcal T_i=\\{t:\\ t\\le K_0,\\ t\\in$ 反向调度$\\}$ 为链 $i$ 实际捕获的桥步集合
（$K_0{=}3$ 为捕获窗的\\emph{时间步阈值}，与递归次数上限 $R_{\\max}{=}2$ 是两个独立参数），
$r_t$ 为该步实际执行的末次递归（自洽提前停机时取停机前残差），$\\tilde x_0^{(0)}\\equiv 0$；
分母为固定 $K_0 N$——步长跳跃使实际捕获步数可少于 $K_0$，此差异如实保留而非调整分母
（公式与代码逐像素一致，经 $4\\times4$ 极小输入人工核对）。直觉：生成器反复修正仍无法自洽的
像素，正是其学到的先验证据不足之处——幻觉候选区。我们将其准确定位为
\\emph{采样器内部收敛不稳定性的代理信号}：递归不收敛也可来自数值迭代或纹理敏感性，
并不严格等价于认知不确定性~\\cite{kendall2017uncertainties}。该信号不需要任何额外生成器
前传（递归本来就要算；缓存与归一化的开销见成本表）。两类信号与采样链的关系见
图~\\ref{fig:overview}。""")

# ================= ZH §3.3 正交→互补 =================
splice(ZH,
       "我们将其与一个非分歧信号（CReF）配对，后者与采样方差正交。",
       "后者与采样方差正交。",
       """我们将其与一个非分歧信号（CReF）配对；两者的像素级 Spearman 相关在 \\S4.4 实测，
"互补"以实测相关性为准而非概念断言。""")

# ================= ZH §3.4 归一化声明 =================
splice(ZH,
       "两图经 99 分位截断的稳健归一化后融合为",
       "默认 $w_1{=}0.6$、",
       """两图经 99 分位截断的稳健归一化后融合为
$T(p)=1-w_1\\widehat{\\mathrm{CReF}}(p)-w_2\\widehat{U}(p)$（$q{=}99$ 为逐图独立计算的
固定算子参数，不从数据分布拟合），默认 $w_1{=}0.6$、""")

# ================= EN §3.2 =================
splice(EN,
       "Within one reverse step, the recursion produces a sequence of pixel-wise",
       "relates both signals to the sampling chain.",
       """Each reverse step produces a pixel-wise residual; SelfRDB keeps only its
global mean as the stopping criterion and discards the spatial information.
We retain the \\emph{last-recursion residual} of each captured reverse step:
\\begin{equation}
  \\mathrm{CReF}(p) \\;=\\; \\frac{1}{K_0\\cdot N}\\sum_{i=1}^{N}\\sum_{t\\in\\mathcal T_i}
      \\bigl\\lvert \\tilde x_0^{(r_t,i)}(p)-\\tilde x_0^{(r_t-1,i)}(p)\\bigr\\rvert ,
  \\label{eq:cref}
\\end{equation}
where $\\mathcal T_i=\\{t:\\ t\\le K_0,\\ t\\in\\text{schedule}\\}$ is the set of bridge
steps actually captured for chain $i$ ($K_0{=}3$ is the \\emph{time-index
threshold} of the capture window, a parameter distinct from the recursion cap
$R_{\\max}{=}2$), $r_t$ is the last recursion actually executed at step $t$ (on
early stopping, the residual just before stopping), and $\\tilde x_0^{(0)}\\equiv 0$;
the denominator is the fixed $K_0 N$ --- with step skipping the captured set can
be smaller than $K_0$, and this discrepancy is kept as-is rather than absorbed
into the denominator (the formula matches the code pixel-for-pixel, verified by
a hand calculation on a $4\\times4$ input). Intuition: pixels the generator keeps
revisiting without converging are those where its learned prior provides
insufficient evidence --- prime candidates for hallucinated content. We position
it precisely as \\emph{a proxy of sampler-internal convergence instability}:
non-convergence can also stem from numerical iteration or texture sensitivity
and is not strictly epistemic uncertainty~\\cite{kendall2017uncertainties}. The
signal requires no extra generator forwards (the recursion runs anyway; caching
and normalization overheads are tabulated in the cost table).
Figure~\\ref{fig:overview} relates both signals to the sampling chain.""")

# ================= EN §3.3 =================
splice(EN,
       "we pair it with a",
       "orthogonal to sampling variance.",
       """we pair it with a non-dispersion signal (CReF); the pixel-level Spearman
correlation between the two is measured in \\S\\ref{sec:exp-uq}, and
"complementary" is used in the measured sense rather than as a conceptual
assertion rather than claiming orthogonality without measurement.""")

# ================= EN §3.4 归一化声明 =================
splice(EN,
       "Both fields are robustly normalized (99th-percentile clipping) and fused as",
       "$w_1{=}0.6$, $w_2{=}0.4$.",
       """Both fields are robustly normalized (99th-percentile clipping; $q{=}99$ is a
fixed per-image operator parameter, not fitted from any data distribution) and
fused as $T(p)=1-w_1\\widehat{\\mathrm{CReF}}(p)-w_2\\widehat{U}(p)$ with
$w_1{=}0.6$, $w_2{=}0.4$.""")

print("patch A2 complete")
