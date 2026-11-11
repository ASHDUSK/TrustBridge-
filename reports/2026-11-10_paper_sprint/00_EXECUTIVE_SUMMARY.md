# 00 · 执行摘要

**审计日期**：2026-11-08 至 2026-11-10
**审计范围**：TrustBridge 论文 v0.1→v0.8.1 全部代码、实验和统计口径
**分支**：audit/v0.8-fixes（本地），main 未受影响

## 核心结论

| 维度 | 判定 |
|---|---|
| AUROC body mask | ❌ **21 个脚本中仅 4 个正确应用**——论文主表全部 AUROC 来自全图路径 |
| 融合权重 | ❌ **sampler 默认与论文互为反转**（0.4/0.6 vs 0.6/0.4）——仅 5 个脚本手动覆盖正确 |
| CReF 公式 vs 代码 | ✅ **逐像素一致**（Gate 1 手算 14/14 PASS） |
| K 消融方向 | ❌ eval_derive.py 取源端步（已修正为数据端） |
| 患者级统计 | ✅ 已完成（B=5000，8 被试配对 bootstrap） |
| 论文 v0.8.1 | 部分——新公式/定义/架构已写入，但**旧 AUROC 数字尚未被修复后数字替代** |

## P0 问题（必须修）

| # | 问题 | 已修复 | 需重算 |
|---|---|---|---|
| 1 | AUROC 未 mask（17/21 脚本） | 代码已修 | **是（全量）** |
| 2 | 融合权重反转（sampler vs 论文） | 部分脚本已显式覆盖 | **是（res.trust_map 用户）** |
| 3 | K 消融取 ts 方向反 | eval_derive.py 已修 | 离线重算即可 |
| 4 | 0.90/0.78 口径演示模型归属混淆 | 论文措辞已修 | 需补跑官方整图口径或删去 |
| 5 | "dominates TTA" 缺 CI 佐证 | 3 处残留 | 需 CReF-N1 vs TTA 配对 CI |

## 实验重算清单

| 实验 | GPU 预计 | 优先级 | 状态 |
|---|---|---|---|
| eval_uq_baselines T1→T2（self ckpt） | ~8 min | P0 | ✅ 已完成（results/2026-11-10_） |
| eval_uq_baselines T2→T1（self ckpt） | ~8 min | P0 | 🔄 运行中 |
| eval_degradation（self ckpt） | ~25 min | P0 | ✅ 已完成 |
| eval_tumor_sparsify（official ckpt） | ~20 min | P0 | 🔄 需重跑（mask+weight 修复后） |
| eval_crossinst_brats（official ckpt） | ~45 min | P0 | 🔄 需重跑 |
| eval_body_decile（self ckpt） | ~15 min | P0 | 🔄 需重跑 |
| eval_paper_extras collapse | ~10 min | P1 | 🔄 需重跑 |
| eval_paper_extras routing | ~10 min | P1 | 🔄 需重跑 |
| eval_paper_extras semantic | ~10 min | P2 | 🔄 需重跑 |

## 可以信任的数字（mask 正确 + 权重正确的路径）

| 数字 | 来源脚本 | 口径 |
|---|---|---|
| 肿瘤定位 0.51±0.10 | eval_tumor_corrected.py | within-body ✓ |
| K 消融 derived.json | eval_derive.py | within-body ✓ |
| TTA 基线 0.42 | eval_tta_baseline.py | within-body ✓ |
| maps_full K=10 AUROC | eval_maps_full.py | within-body ✓ |
| 体区十分箱 | eval_body_decile.py | within-body ✓（但用 sampler 权重 0.4/0.6） |

## 不能信任的数字（需重算）

| 数字 | 原因 |
|---|---|
| 表2 全部 AUROC（0.900/0.816 等） | eval_uq_baselines 全图口径 |
| 跨机构 0.950/0.863 | eval_crossinst_brats 全图口径 |
| 退化响应 AUROC 列 | eval_degradation 全图口径 |
| BraTS 0.911 | eval_tumor_sparsify 全图口径 |
| Routing 0.880 | eval_paper_extras 全图口径 |

## 明早最优先的三件事

1. **等重算完成后**：用新 AUROC 数字替换论文全部表格和正文引用
2. **患者级 CI 回填**：从 eval_patient_stats/summary.json 提取全部 CI 写入论文
3. **编译双 PDF + visual-judge**：确认新旧数字无混用
