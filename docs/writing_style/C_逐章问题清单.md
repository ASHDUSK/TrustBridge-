# C · 当前论文逐章问题清单（v0.8.4 → v0.8.5）

> 审查对象：main.tex / main_zh.tex（v0.8.4，c4c2bbe）。问题分级：【P1 主张-证据/口径】【P2 结构错位】【P3 项目报告/宣传痕迹】【P4 表达】。

| # | 位置 | 原文（节选） | 问题 | 级别 | 参考规范 | 修改 |
|---|---|---|---|---|---|---|
| 1 | 标题副题（EN:29/ZH:20） | "(system: TrustBridge --- a training-free trust layer...)" | 产品化命名入标题（规范 §6） | P3 | SelfRDB 标题无系统名 | 去 "system:"，保留 TrustBridge 命名 |
| 2 | 摘要（EN:41-76） | 15 个数值 | 数字密度远超领域惯例（参考摘要 0 个） | P4 | A §1 | 修剪至 ≤10：删单链 0.79/0.70、官方 0.57/0.63、像素 AUROC 0.61/0.82 三处冗余 |
| 3 | 摘要 EN:53 | "we harvest two training-free signals" | 工程动词 | P3 | 规范 §6 | harvest → extract |
| 4 | 摘要 EN:74 / ZH | "as a boundary map for the field" | 隐喻性宣传 | P3 | 规范 §6 | "as negative results delimiting the applicability of such methods" |
| 5 | 引言 EN:110-111/ZH | "(§4.6)" 硬编码节号 | 交叉引用不规范 | P4 | — | 改 \S\ref |
| 6 | 相关工作表 tab:rw caption | "The "no extra" entry of The "no extra" entry refers to..." | v0.8.4 引入的重复残句 | P4 | — | 删残句 |
| 7 | §3.2（EN:287/ZH:185） | "(demo defaults, uncalibrated)" | 项目演示语言 | P3 | 规范 §6 | "(default values, not clinically calibrated)" |
| 8 | §3.2 ZH:185 | "与 UI 叠加图一致" | UI 引用入方法 | P3 | 规范 §6 | 删 UI 引用 |
| 9 | §4.2（EN:343） | "reproduction checklist (docs/18, with...)" | 内部文档路径 | P3 | 规范 §6 | "a documented reproduction checklist with verified/unverified status" |
| 10 | §4.2（EN:356-358） | "The deployment implication is starker for being graded: ...a 20-point trust gap is a signal, not a gate." | 口号句+教训句在结果章 | P2/P3 | 规范 §3/§6 | 改限定式结论并去口号 |
| 11 | §4.8（EN:~594） | "The lesson goes beyond our method: ...must be audited against pathology annotations..." | 讨论内容占结果段尾 | P2 | 规范 §3 | 压缩为一句机制概括+\S5 交叉引用 |
| 12 | §4.9（EN:753/ZH:74,497） | "we publish our retraction trail (docs/18 and the corrected analysis scripts)" | 内部路径+publish 措辞 | P3 | 规范 §6 | "we provide the retraction trail and corrected analysis scripts as evidence..." |
| 13 | §4.9 ZH:218 | "均已记录在复现排查表中（docs/18，区分...）" | 同上 | P3 | 规范 §6 | 同 12 的处理 |
| 14 | 讨论（EN:212-214 fig1 caption） | "are harvested from the recursion" | 工程动词 | P3 | 规范 §6 | extracted |
| 15 | 结论（EN:795） | "harvesting it provides..." | 同上 | P3 | 规范 §6 | deriving |
| 16 | §2.2 ZH 相关工作 | （对照表完整、有定位句） | 质量达标，无需重构 | ✓ | — | 仅随 #6 修表注 |
| 17 | §4.4/§4.5/§4.7/§4.10 | 数字、口径、统计三件套 | v0.8.3/4 已达标 | ✓ | — | 不动 |
| 18 | §5 讨论 | 条件清单/两层架构/局限 | 已达标（含 RAISE 式 guardrail 语言） | ✓ | — | 不动 |
| 19 | 摘要 ZH 对应 #2/#4 | 数字密度/边界地图隐喻 | 同 EN | P4 | 同上 | 镜像修改 |

未发现的问题类型（对照任务清单 1-8）：无"项目功能介绍代替科学论证"（§4.6 的 QC 负结果本身是科学结论）；无编造统计；无文献罗列（相关工作有比较分析）；变量/公式定义在 v0.8.3 后完整。
