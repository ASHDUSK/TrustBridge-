# 02 · 融合权重审计报告（Agent B）

## 核心结论

代码默认融合权重（CReF=0.4, Ensemble=0.6）与论文声称（CReF=0.6, Ensemble=0.4）**互为反转**。论文口径只存在于 5 个手动重算脚本中；全部走 `res.trust_map` 的端到端实验用的是反转权重。

## 权重定义点

| 位置 | CReF 权重 | Ens 权重 | 与论文关系 |
|---|---|---|---|
| sampler.py:61-62 `__init__` 默认 | 0.4 | 0.6 | **相反** |
| sampler.py:49-54 `preset_for()` 非官方 CT | 0.4 | 0.6 | **相反** |
| sampler.py:49-54 `preset_for()` 官方 CT | 0.8 | 0.2 | 不同 |
| sampler.py:34 `TrustResult.w_used` 默认值 | 0.6 | 0.4 | 表面一致，实际被覆盖 |
| pipeline.py:10-15 `get_sampler()` | 按 preset_for | | 同上 |

## 配置矩阵

| 脚本/入口 | trust 来源 | CReF | Ens | 与论文一致？ |
|---|---|---|---|---|
| eval_uq_baselines.py ours_fused | 手动 0.6R+0.4U | **0.6** | **0.4** | ✅ |
| eval_degradation.py | res.trust_map | 0.4 | 0.6 | ❌ 反转 |
| eval_tumor_sparsify.py | res.trust_map | 0.4 | 0.6 | ❌ 反转 |
| eval_crossinst_brats.py | res.trust_map | 0.4 | 0.6 | ❌ 反转 |
| eval_body_decile.py | res.trust_map | 0.4 | 0.6 | ❌ 反转 |
| eval_maps_full.py | res.trust_map | 0.4 | 0.6 | ❌ 反转 |
| eval_derive.py --w 0.6 | 手动 0.6R+0.4U | **0.6** | **0.4** | ✅ |
| ablation_trust.py full_n4 | 手动 0.6R+0.4U | **0.6** | **0.4** | ✅ |
| app/app.py + api.py | res.trust_map | 0.4/0.8 | 0.6/0.2 | ❌ |
| evaluate_your_model.py | res.trust_map | 0.4 | 0.6 | ❌ 反转 |

## 关键发现

1. **sampler 全链路（默认与 preset_for）实际用 CReF=0.4 / Ens=0.6**，论文声称 0.6/0.4——互为反转。
2. **论文口径只存在于 5 个手动重算脚本**，而全部走 `res.trust_map` 的端到端实验（degradation、crossinst、tumor、decile、collapse、maps_full、应用/UI）用的是反转权重。
3. **双重归一化**：手动路径对 `res.rcrf_map`/`res.unc_map`（sampler 已归一化）再做 `_pct_norm`。
4. **官方 CT preset 0.8/0.2** 只对精确 task_id 生效。
5. **TrustResult.w_used 默认值 (0.6,0.4)** 让未覆盖字段的消费方显示错误值。
6. **文档矛盾**：docs/12 和 docs/10 称 w_residual=0.6——与代码默认 0.4 冲突。

## 影响

受反转权重影响的论文数字：
- §4.9 退化响应（trust 值来自 sampler 0.4/0.6）
- §4.8 跨机构（trust 来自 sampler 0.4/0.6）
- §4.5 体区十分箱（trust 来自 sampler 0.4/0.6）
- 崩塌甄别（trust 来自 sampler 0.4/0.6）
- 应用端全部 trust_map 显示

不受影响的（手动 0.6/0.4）：
- eval_uq_baselines ours_fused（手动覆盖）
- eval_derive.py --w 0.6（手动覆盖）
- ablation_trust.py（显式传参）
