# 03 · 实验结果总审计（Agent C + 全量溯源）

## 1. sampler 递归停止条件

### 精确数学表达

每个 chunk（bs 条链）每个反向步 t：

```
x0_r ← 0                                            # x̃0^(0) ≡ 0
for r in 1..R_max:                                   # R_max = 2
    x0_rp1 ← G_θ([x_t, y], t, x_r = x0_r)
    res_last ← |x0_rp1 − x0_r|.mean(dim=1)           # [bs,H,W] 通道均值
    change ← max_pixels( mean_chains(res_last) )      # 标量：链均值后像素 max
    if change < τ = 0.01: break                       # batch 级停机
    x0_r ← x0_rp1

x0_pred ← x0_rp1 if r == 1 else x0_r                 # 防零图 / 取倒数第二估计
```

### 停止粒度

- **batch/chunk 级**：change 是单个标量，链不能各自停
- 判据：先链均值、再像素 max → 比全局均值更严格
- 与官方 SelfRDB diffusion.py L98 逐字等价
- **论文 §3.1 描述不准**：写的是"全局均值 1%"，实际是 max-of-batch-mean

### x0_pred 语义

| 情形 | x0_pred | 说明 |
|---|---|---|
| r=1（仅一次递归） | x0_rp1 | 防止零图（TrustSampler 改进） |
| r≥2 早停 | x0_r（倒数第二估计） | 最新估计被丢弃，残差已入 CReF |
| r≥2 耗尽 | x0_r == x0_rp1 | 最新估计 |

实践中 r 恒为 2（阈值 0.01 下 r=1 早停需全图幅值 <1%）。

### 论文 §3.1 需修正

"global mean 1%" → "max of batch-mean |Δx₀| < 1%"（更严格）

## 2. CReF 分母与 step_skip

### 代码语义

```
rcrf_acc = Σ_{chains} Σ_{t ∈ captured_steps} D_t(i)
CReF_raw(p) = rcrf_acc(p) / (K₀ × N)                # 固定分母
```

- K₀ = capture_last_steps = 3（时间步阈值，非递归次数）
- N = n_ensemble 总链数
- captured_steps = {t ∈ schedule : t ≤ K₀}
- skip=1: |captured|=3 = K₀ → 一致
- skip=2: |captured|=2 < K₀=3 → CReF_raw 被缩小 2/3 倍
- **缓解**：pct_norm 逐图除以自身 99 分位 → 均匀缩放被精确抵消 → 排序指标不受影响
- **论文已声明**：main.tex L183-185 "discrepancy is kept as-is"

### 与论文一致性

| 论文断言 | 代码 | 一致？ |
|---|---|---|
| 末次递归残差 | res_last 是 break 前计算的 | ✅ |
| 固定分母 K₀·N | L242 精确匹配 | ✅ |
| step_skip 差异如实保留 | 已声明 | ✅ |
| x̃₀^(0)≡0 | L192 零初始化 | ✅ |
| 手算核对 | gate1_verify.py 14/14 PASS | ✅ |

**不一致项**：论文 §3.1 "global mean 1%" → 应为 "max of batch-mean |Δx₀| < 1%"

## 3. K 消融实现审计

### 发现方向性 bug

`eval_derive.py` L107:
```python
keep_ts = set(sorted(ts, reverse=True)[:K])   # 取最大 K 个 t → {10,9,8}
```

应为：
```python
keep_ts = set(sorted(ts)[:K])                  # 取最小 K 个 t → {1,2,3}
```

**影响**：6 组 derived.json 的 K<10 行实际度量的是"源端 K 步窗"（靠近 y_ε），
不是论文定义的"数据端捕获窗"（靠近 x₀）。K=10 不受影响（全部步）。

**修复**：eval_derive.py 已改为 `sorted(ts)[:K]`。6 组 derived.json 需重跑离线派生（无需 GPU，纯 npz 重算）。

### K=1 特殊语义

K=1 时只保留 t=1 的残差。t=1 时 x0_r 初始为 0，故残差 = |生成器输出本身|。
这与 K≥2 的"相邻估计差"语义不同——文中需注明。

## 4. 健康模型退化实验数值矛盾

### 已确认的两份结果

| 来源 | none trust | blur trust | 模型 |
|---|---|---|---|
| results/2026-11-03_degradation_response/ | 0.630 | 0.703 | 官方 T1→T2（12.98 dB） |
| results/2026-11-09_degradation_response/ | 0.831 | 0.834 | 自训 T1→T2（18.24 dB） |

**结论**：不矛盾——是两个不同模型的独立实验。0.63 是官方权重（崩塌基座），
0.83 是自训权重（健康基座）。论文应明确标注模型身份。

### 信任差异解释

- 健康模型基线信任 0.831（更高，因为翻译更好）
- 崩塌模型基线信任 0.630（更低，因为输出质量差）
- 两者对模糊的响应方向一致（均升高），但崩塌模型升高更多
- maha 对两者一致（2.9→15.0），因为 maha 度量输入空间性质与模型无关

## 5. 交叉引用审计

| 引用 | 指向 | 正确？ |
|---|---|---|
| §4.1 "阈值敏感性见 §4.9" | §4.9 是退化实验，无阈值敏感性 | ❌ 应删或补做 |
| §3.3 "互补性见 §4.4" | §4.4 有 U 表但无 Spearman 数值 | ⚠ 需补 CReF-U ρ |
| §3.2 "手算核对" | gate1_verify.py 14/14 PASS | ✅ |
| §4.2 "docs/18 排查表" | docs/18 存在 | ✅ |
