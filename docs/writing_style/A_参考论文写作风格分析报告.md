# A · 参考论文写作风格分析报告

> 样本（10 篇，逐篇抽取摘要/引言/方法/实验/讨论关键页）：
> ①SelfRDB（MedIA 2025，同任务基座）②MUMINS（arXiv 2026，不确定性感知合成）
> ③Wolleb（MIDL 2022，扩散集成不确定性）④Osuna-Vargas（WACV 2025，显微去噪）
> ⑤RAISE（Cardoso 等，放射 AI 安全生命周期）⑥Measurement Geometry（arXiv 2026，
> 幻觉的测量几何理论）⑦Hallucination Index（MICCAI 2024）⑧DB-3D（arXiv 2025，
> 桥式 3D 翻译）⑨3D 超声扩散重建（IEEE TMI 2025）⑩（life-15-00654 为⑥重复副本）。

## 1. 摘要的共同骨架

| 功能句 | SelfRDB 实例 | MUMINS 实例 |
|---|---|---|
| 背景 1–2 句 | "DDMs have gained recent traction..." | "Forecasting anatomical changes ... is a challenging generative vision task" |
| 现有方法的具体失效 | "diverges significantly ... weak source-modality guidance" | "deterministic networks ignore biological stochasticity; standard diffusion models require computationally prohibitive multi-pass sampling" |
| "Here we propose..." + 机制 2–4 句 | novel forward process / noise scheduling / recursive estimation | joint baseline-residual / re-injection / NLL head |
| 评价范围 + **克制的比较级结论** | "indicate that SelfRDB offers superior performance against competing methods" | "matches or outperforms dedicated, domain-specific state-of-the-art methods" |

**关键规范**：摘要**几乎不放具体数字**（10 篇中 0 篇在摘要报主指标数值）；结论用比较级/限定词
（superior/matches or outperforms/better or similar），绝不用"全面领先/唯一"。

## 2. 引言的缺口推进结构

Clinical motivation（带引文）→ 问题的形式化难度（"heavily ill-posed"、"hard conditional
generation problem"）→ 方法族演进，**每族一句带引文的局限**（GAN 不稳定→DDM 任务无关→桥
未探索）→ 明确的 gap 句（SelfRDB："remains largely unexplored"；Hallucination Index：
"the field lacks a standard methodology"；Osuna："it is currently unclear how well..."）→
"Here, we propose/demonstrate..." → Contributions 三条目。

**修辞特征**：允许一句强主题句（SelfRDB："Trust, however, has not kept pace with fidelity"
级别的转折句在好论文中常见），但紧随其后立刻回到证据语气。

## 3. 相关工作 = 按"方法族"组织，每族以局限收尾，最后定位

SelfRDB 的 Related Work 每段都以"Unlike X, SelfRDB ..."式定位句结束；MUMINS 按子领域分节、
节内列举+归纳失效。**不罗列文献**——每篇被引文献都承担"证据某主张/某局限"的角色。

## 4. 方法章节

符号一次性定义+沿用（MUMINS："For brevity, we henceforth write ... suppressing
dependence on ... unless required for clarity"）；公式编号且每个新符号即定义；假设显式
（Measurement Geometry："Under local regularity assumptions, we prove..."）。

## 5. 结果与讨论的边界

结果章节允许**一句**解读性收束，但系统性的"lessons/implications"集中在 Discussion；
失效分析在结果中给数据、在讨论中给机制。RAISE（立场论文）展示讨论语言的安全边界：
"no single AI solution can provide absolute assurance even when limited to its intended use"、
"Trust in AI is an earned privilege"——克制但有立场。

## 6. 措辞强度谱系（可复用的限定词库）

- 存在性弱断言：can suffer / can hamper / may be well-suited / tends to
- 比较级结论：superior against competing methods / matches or outperforms / better or
  similar to / consistently outperforms
- 证据动词：indicate / suggest / demonstrate（demonstrate 仅用于有直接数据支持处）
- 回避：prove（除定理）、novel 滥用（每篇最多 2–3 处）、第一/唯一（SelfRDB 敢用
  "first ... to our knowledge"——有文献检索支撑时可用）

## 7. 项目报告式表达的判定特征（TrustBridge 需要清除的）

- 引用内部文档路径（docs/18）而非论文级产物（Supplementary / released scripts）
- 产品化命名（"系统：TrustBridge"入标题；UI 引用入方法章节）
- 工程动词（harvest/收割、拆成两层）
- 口号式对句（"a signal, not a gate"；"知道自己不知道"）
- 摘要数字密度超标（对照：参考论文摘要 0 个主指标数字）

## 8. 各论文之间的差异（防止单篇偏好）

- 数字密度：TMI/WACV 类在结果正文用表；摘要均无数值。
- 立场论文（RAISE）允许更强的规范性语言（should/imperative），方法论文不允许。
- 理论文（Measurement Geometry）可用 prove/theorem；实证文用 indicate/demonstrate。
- 结论：写作规范必须取**交集**（所有论文共同遵守的），而非任何单篇的个性。
