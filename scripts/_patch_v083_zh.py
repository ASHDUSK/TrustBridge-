# -*- coding: utf-8 -*-
"""v0.8.3 ZH 正文补丁：数据集勘误 + 阈值敏感性 + 参数表 + §4.4 敏感性段 + SynthRAD 小节。"""
import io

P = r"D:\MedTranslateProject\docs\latex\main_zh.tex"
s = io.open(P, encoding="utf-8").read()


def rep(old, new, tag):
    global s
    assert old in s, f"[{tag}] NOT FOUND: {old[:80]!r}"
    assert s.count(old) == 1, f"[{tag}] NOT UNIQUE: {old[:80]!r}"
    s = s.replace(old, new, 1)


# ---------- 1. 数据集：SynthRAD 3 例 ----------
rep(r"""T2 真值与肿瘤分割，源 FLAIR。\textbf{SynthRAD2023 盆腔}~\cite{thummerer2023synthrad}：
10 例，中心匹配 T1w$\to$CT 与 T2w$\to$CT，各 80 测试切片。""",
    r"""T2 真值与肿瘤分割，源 FLAIR。\textbf{SynthRAD2023 盆腔}~\cite{thummerer2023synthrad}：
3 例（T1w$\to$CT 2 例、T2w$\to$CT 1 例；128$\times$128 切片，分别 160 与 80 个，
即全部已制备切片）——是跨解剖的试点检验，而非总体样本（\S\ref{sec:exp-synthrad}）。""",
    "datasets")

# ---------- 2. 阈值敏感性 ----------
rep(r"""$[0,1]$ 图；无形态学后处理——简化协议，阈值 0.01/0.05 的敏感性见 \S4.9）。正类=该片体区内""",
    r"""$[0,1]$ 图；无形态学后处理——简化协议）。掩码阈值敏感性在脑数据上很小（体区 AUROC 在阈值
$0.01/0.02/0.05$ 下分别为 $0.830/0.822/0.817$（T1$\to$T2）与 $0.753/0.739/0.737$（T2$\to$T1）），
在跨解剖数据上更大（\S\ref{sec:exp-synthrad}）。正类=该片体区内""",
    "threshold")

# ---------- 3. 参数表 ----------
rep(r"""与 BraTS FLAIR$\to$T2 评估。
\paragraph{信任指标口径：}""",
    r"""与 BraTS FLAIR$\to$T2 评估。
\paragraph{实现与分析配置：}
表~\ref{tab:impl} 冻结本文全部数字所用的唯一配置；以下参数均不逐实验调优，
偏离该配置之处一律以敏感性分析形式报告（\S\ref{sec:exp-uq}、\S\ref{sec:exp-synthrad}）。
\begin{table}[t]
\centering\small
\caption{实现与分析配置（全部实验统一；推理种子 2026）。}
\label{tab:impl}
\begin{tabular}{p{0.24\linewidth}p{0.68\linewidth}}
\toprule
组件 & 设置 \\
\midrule
基座模型 & SelfRDB 桥，10 个扩散步（$\beta$：$0.1\to3.0$） \\
自训模型 & 128$\times$128，30 epochs，batch 8，lr $1.6\cdot10^{-4}$/$10^{-4}$（G/D），Adam $(0.5,0.9)$，bf16-mixed，seed 42，单卡 RTX 4060 Laptop 8\,GB \\
采样 & $N{=}4$ 独立链，递归上限 $R_{\max}{=}2$，步长跳跃 1 \\
捕获窗 & $K_0{=}3$ 个数据端桥步（$\mathcal T_i{=}\{t\le K_0\}$） \\
归一化 & 逐图 99 分位截断到 $[0,1]$（固定算子） \\
融合 & $T = 1-(0.6\,\mathrm{CReF}_n + 0.4\,U_n)$；权重先验设定 \\
等预算 & 全部 $N{=}4$ 分歧信号统一 40 次生成器前传 \\
图像级评分 & Trust Score $=100\times$ 体区均值；PASS/REVIEW 门限 75/55（演示默认，未标定） \\
输入门 & 6 类模态分类头 + 逐类 Mahalanobis（$\tau{=}389$，门自留训练集拟合）+ 平坦图规则 \\
统计口径 & 逐片指标宏平均；患者级配对 bootstrap $B{=}5000$ \\
\bottomrule
\end{tabular}
\end{table}
\paragraph{信任指标口径：}""", "impl")

io.open(P, "w", encoding="utf-8").write(s)
print("ZH patches 1-3 OK")
