# -*- coding: utf-8 -*-
"""论文 v0.4 patch 4：崩盘甄别真相（结构性崩塌+守卫失效）、语义代理阴性、QC r 值、贡献点。"""
import io

fp = r"D:\MedTranslateProject\docs\latex\main.tex"
s = io.open(fp, encoding="utf-8").read()

# ---- 1) 复现章第二段重写：结构性崩塌 + 守卫失效 ----
i0 = s.index("\\paragraph{What the trust layer does and does not see in a collapse.}")
i1 = s.index("\n\n", s.index("future work).", i0))
new_par = r"""\paragraph{What the trust layer does and does not see in a collapse.}
A natural hope is that sampler-native signals would ``know'' the base model
is failing. We tested this directly, and the answer is the most important
negative result in this paper. On the collapsed official model, the trust
map does not fire---mean trust $94.2/100$, low-trust ratio $5.9\%$ (lower
than for our healthy self-trained model), all 20 test slices PASS. And the
collapse is \emph{not} a benign photometric offset: within-body NCC is
$0.41$, SSIM $29\%$, and a ground-truth-fitted linear intensity correction
leaves PSNR at $12.3$\,dB---the official model synthesizes
\emph{structurally wrong but internally self-consistent} anatomy. The
sampler is confident about its wrong output, so CReF and dispersion---which
measure self-consistency---are blind, exactly as our tumor analysis
predicted at the pixel level, now reproduced at the model level. The lesson
for deployment is stark: \emph{training-free internal signals cannot detect
a base model that is confidently wrong}. They complement, but do not
replace, external validation against ground truth; a guard layer that has
never seen its base model fail cannot be trusted to recognize failure."""
s = s[:i0] + new_par + s[i1:]

# ---- 2) 摘要中的错误声明替换 ----
old = ("A dedicated reproducibility analysis\n"
       "explains why officially released weights collapse to ${\\sim}15$\\,dB on\n"
       "independently prepared data (paper value 31.56\\,dB), and shows the trust map\n"
       "flagging the collapse globally without ground truth.")
assert old in s, "abstract repro claim"
new = ("A dedicated reproducibility analysis\n"
       "explains why officially released weights collapse to ${\\sim}15$\\,dB on\n"
       "independently prepared data (paper value 31.56\\,dB)---structurally, not\n"
       "just photometrically (NCC 0.41)---and shows that the trust layer\n"
       "\\emph{fails to notice}: the collapsed model receives PASS with $94/100$\n"
       "mean trust. Self-consistency cannot certify correctness; external\n"
       "validation remains irreplaceable.")
s = s.replace(old, new)

# ---- 3) §4.6 r 值更新为自训模型 + BraTS ----
old = ("correlates with slice quality (body PSNR) weakly and \\emph{directionally\n"
       "inconsistently}: Pearson $r=-0.22$ (T1$\\to$T2), $r=+0.33$ (T2$\\to$T1),\n"
       "$r=-0.10$ (BraTS).")
assert old in s, "qc r values"
new = ("correlates with slice quality (body PSNR) weakly: Pearson $r=+0.47$\n"
       "(T1$\\to$T2, self-trained), $r=+0.26$ (T2$\\to$T1, self-trained),\n"
       "$r=-0.10$ (BraTS, official)---right-signed on the models whose signals\n"
       "work, absent on the one whose do not.")
s = s.replace(old, new)

# ---- 4) 边界实验加第 (7) 项：语义代理阴性 ----
old = ("(6)~\\emph{T2$\\to$T1 signal failure on the official weights}\n"
       "(\\S\\ref{sec:exp-uq}): all dispersion signals at chance when that model's\n"
       "residual error is texture-dominated---while the same direction works on our\n"
       "self-trained model ($0.816$), isolating the failure to the model, not the task.")
assert old in s, "boundary 6"
new = old + ("; and (7)~\\emph{semantic-proxy signals do not complement dispersion}:\n"
             "block-wise normalized mutual information and gradient-orientation\n"
             "agreement between source and synthesis reach only AUROC\n"
             "$0.42$--$0.67$ where dispersion reaches $0.82$--$0.90$, and their\n"
             "combination with dispersion does not improve it---structural\n"
             "agreement with the \\emph{source} is a weaker error surrogate than\n"
             "sampler-native dispersion.")
s = s.replace(old, new)

# 边界实验计数 six -> seven（§4.7 引导句与贡献点3）
s = s.replace("Six controlled experiments delineate", "Seven controlled experiments delineate")
s = s.replace("We report six controlled experiments", "We report seven controlled experiments")
s = s.replace("Six controlled\n        experiments", "Seven controlled\n        experiments")

# ---- 5) 贡献点3 的 T2→T1 表述定位到官方权重 ----
old = ("round-trip consistency, and the\n        T2$\\to$T1 signal failure")
if old in s:
    s = s.replace(old, ("round-trip consistency, the\n        official-weights T2$\\to$T1 signal failure, and semantic-proxy signals"))
io.open(fp, "w", encoding="utf-8").write(s)
print("patch 4 done")
