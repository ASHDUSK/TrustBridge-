# 01 · AUROC 体掩码审计报告（Agent A）

## 核心结论

论文声明的体掩码协议（source>0.02）与代码实现不符。全部 21 个含 AUROC 的脚本中，只有 4 个真正在掩码内计算 AUROC。论文主表引用的 AUROC 数字全部产自全图（含空气像素）代码路径。

## 根因

共有 6 份独立的 `auroc_low_vs_higherr` / `auroc` 实现（内容相同，秩和法），**没有任何一份接收 mask 参数**。是否正确完全取决于调用方是否预先过滤了数组。

## 逐脚本审计表

| 脚本 | mask 应用 | 影响的论文数字 |
|---|---|---|
| eval_uq_baselines.py | ❌ | §4 UQ 表全部 AUROC |
| ablation_trust.py | ❌ | docs/08 消融表 |
| eval_degradation.py | ❌ | 退化响应 AUROC |
| eval_tumor_sparsify.py | ❌ | BraTS 0.911 |
| eval_crossinst_brats.py | ❌ | 跨机构 0.863/0.950 |
| eval_paper_extras.py | ❌ | collapse/routing/semantic |
| eval_gt_trust.py | ❌ | Tab.1/3 来源 |
| trust_vs_tumor.py | ❌ | 0.871（已撤回） |
| cycle_signal.py | ❌ | 回译 CCR |
| learned_fusion.py | ❌ | learned_fusion 表 |
| eval_bf16.py | ❌ | bf16 对照 |
| validate_trust.py | ❌ | 早期原型 |
| evaluate_your_model.py | ⚠ auroc_body 与 auroc_whole_image 调用同一函数 | 口径对照失效 |
| **eval_derive.py** | **✅** | maps_full derived.json |
| **eval_tumor_corrected.py** | **✅** | 0.51 撤回数 |
| **eval_brats_trust.py** | **✅** | FeTS 0.5099 |
| **eval_tta_baseline.py** | **✅** | TTA 基线 0.42 |
| **probe_cascade.py** | **✅** | 级联探针 |

## 关键细节

1. `eval_uq_baselines.py::auroc_low_vs_higherr(trust, err)`：trust/err 是全图数组；docstring 写"体区内"但实现无 mask。
2. `eval_predictor()`：spearman 在 mask 内 ✓；auroc 未传 mask ❌；dpsnr/decile 在 mask 内 ✓——同一张 CSV 里 spearman 与 auroc 口径不一致。
3. 正类定义：全图脚本的正类是全图误差 top-10%。空气像素误差近零且信任高（负类），虚高 AUROC。
4. 实证：centerA 与 centerA_masked 的 auroc_fused 完全相同（0.6443），证明 mask 参数不影响 AUROC。

## 修复定位

最小修复 = 在每个调用点将 trust 和 err 过滤为 `[mask]` 子集后再传入。
`eval_derive.py`、`eval_tumor_corrected.py`、`eval_brats_trust.py`、`eval_tta_baseline.py`、`probe_cascade.py` 无需改动（已是正确实现）。

## 额外发现

论文内部口径演示数字自相矛盾：§4.1 "同一融合信号体区 0.78、全图 0.90"——审计显示 0.78 来自 2026-11-03（官方权重体区），0.90 来自 2026-11-04（自训权重全图），**不是同一信号的两种口径**。代码库中不存在任何能产出"体区内 0.78"的官方权重路径。

`evaluate_your_model.py` 的 `auroc_body` 与 `auroc_whole_image` 调用同一无 mask 函数，两列数值恒等。
