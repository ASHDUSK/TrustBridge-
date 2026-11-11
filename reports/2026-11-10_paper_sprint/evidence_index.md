# 08 · 证据索引

## 论文数字 → 结果文件 → 脚本 → 配置

| 论文数字 | 结果文件 | 生成脚本 | 关键参数 | 口径 | 可信度 |
|---|---|---|---|---|---|
| 融合 AUROC 0.900/0.816 | 2026-11-04_uq_baselines_*/summary.json | eval_uq_baselines.py | self ckpt, N=4, seed=2026, w=0.6/0.4 | ⚠ 全图（mask 修复后待重算） | B |
| 融合 AUROC 0.783 | 2026-11-03_uq_baselines_T1_T2/summary.json | eval_uq_baselines.py | official ckpt, N=4, seed=2026 | ⚠ 全图 | B |
| CReF 单链 0.880/0.792 | 同上 ours_rcrf_n1 行 | 同上 | N=1, 10 前传 | ⚠ 全图 | B |
| TTA 0.808/0.482 | 同上 tta_disp 行 | 同上 | 4 增广 × N=1 | ⚠ 全图 | B |
| 噪声 0.763/0.505 | 同上 noise_disp 行 | 同上 | σ=0.02×4 | ⚠ 全图 | B |
| 梯度 0.225/0.578 | 同上 gradient 行 | 同上 | 无前传 | ⚠ 全图 | B |
| 肿瘤定位 0.911±0.037 | 2026-11-03_tumor_sparsify_body/summary.json | eval_tumor_sparsify.py | official ckpt, 60 片, N=4 | ⚠ 全图 | B |
| 肿瘤撤回 0.51±0.10 | 2026-10-31_tumor_trust_corrected/summary.json | eval_tumor_corrected.py | official ckpt, 60 片 | ✅ 体区 | A |
| 肿瘤 1.60× | 同上 err_ratio | 同上 | | ✅ 体区 | A |
| 跨机构 B 0.863±0.155 | 2026-11-04_crossinst_brats/summary.json | eval_crossinst_brats.py | official ckpt, 14 被试 280 片 | ⚠ 全图 | B |
| 跨机构 C 0.950±0.021 | 同上 | 同上 | official ckpt, 21 被试 420 片 | ⚠ 全图 | B |
| K 消融 0.811→0.818 | maps_full/ixi_test_t1t2/derived.json | eval_derive.py | self ckpt, 60 片, mask ✅ | ✅ 体区 | A |
| AUPRC 0.323/0.259 | 同上 | eval_derive.py | | ✅ 体区 | A |
| AUSE 0.014/0.031 | 同上 | eval_derive.py | | ✅ 体区 | A |
| CReF-U ρ 0.721/0.354 | 同上 | eval_derive.py | pooled pixel | ✅ 体区 | A |
| 体区十分箱 193.9×/4.0× | 2026-11-08_body_decile/summary.json | eval_body_decile.py | self ckpt, w=sampler 0.4/0.6 | ✅ 体区 | A |
| 崩塌甄别 PASS 94.2 | 2026-11-04_collapse_alarm/summary.json | eval_paper_extras.py collapse | official ckpt, 20 片 | ⚠ 全图 trust | B |
| 崩塌 NCC 0.41 | 2026-11-04_collapse_ncc/summary.json | eval_collapse_ncc.py | official ckpt, 20 片 | ✅ 体区 | A |
| 退化响应 maha 2.9→15.0 | 2026-11-03_degradation_response/summary.json | eval_degradation.py | official ckpt, 30 片 | ✅ 体区 trust | A |
| 退化响应（健康）trust 0.831→0.834 | 2026-11-09_degradation_response/summary.json | eval_degradation.py --ckpt self | self ckpt, 30 片 | ✅ 体区 trust | A |
| 门控 0/562 | 2026-11-01_gate_v2/summary.json | train_gate_v2.py eval | 6 类分类器+Maha | 不涉及像素 | A |
| 患者级 CI 全部 | 2026-11-09_patient_stats/summary.json | eval_patient_stats.py | 患者级 bootstrap B=5000 | 继承上游 | B |
| 互补性 ρ 0.777/0.409 | 2026-11-09_signal_complementarity/summary.json | analyze_signal_complementarity.py | 患者级+逐片 | ✅ 体区 | A |
| 权重扫描 val 0.787-0.802 | maps_full/ixi_val_*/derived.json | eval_derive.py --w | self ckpt, 6 被试 | ✅ 体区 | A |
| Routing 阴性 | 2026-11-04_routing_cref_tta/summary.json | eval_paper_extras.py routing | self ckpt, 30 片 | ✅ 体区 | A |
| 语义代理阴性 | 2026-11-04_semantic_proxy/summary.json | eval_paper_extras.py semantic | self ckpt, 30 片 | ✅ 体区 | A |
| 盆腔 T1w→CT 24.36 | 2026-09-27_CT_T12CT_self/summary.json | eval_gt_trust.py | self_ct_t12ct, 80 片 | 整图 | A |
| BraTS 24.06±1.62 | 2026-09-28_BRATS_FLAIR2T2_brats2018/summary.json | eval_gt_trust.py | official, 60 片 | 整图 | A |

**可信度等级**：
- **A**：原始数据和脚本均可追溯，口径正确
- **B**：脚本可追溯，但 AUROC 口径为全图（与论文声称的体区不一致）
- **C**：仅论文数字，无法追溯
- **D**：口径错误，需重算

**注意**：eval_derive.py 的 K 消融在 v0.7 中取了源端步（bug 已修为数据端），需重跑离线派生后 K<10 的数字才会正确。
