# 07 · 可复现性检查

## 代码-公式-手算一致性

| 检查项 | 状态 | 证据 |
|---|---|---|
| CReF 公式 Eq.(1) 与 sampler.py 逐像素一致 | ✅ | gate1_verify.py 14/14 PASS |
| 手算 4×4 输入与代码输出一致 | ✅ | gate1_verify.py |
| K₀ 语义（时间步阈值）与代码一致 | ✅ | sampler.py L214 `t <= capture_last_steps` |
| 固定分母 K₀·N 与代码一致 | ✅ | sampler.py L242 |
| x̃₀^(0)≡0 与代码一致 | ✅ | sampler.py L192 |
| 融合公式 T(p)=1-w₁ĈReF-w₂Û 与代码一致 | ✅ | sampler.py L238 |

## 已知不一致（需处理）

| 项 | 论文说法 | 代码实际 | 影响 |
|---|---|---|---|
| 停机判据 | "全局均值 1%" | max(链均值\|Δx₀\|) < 1% | 论文措辞需改 |
| 融合权重 | w₁=0.6/w₂=0.4 | sampler 默认 0.4/0.6 | 部分脚本已显式覆盖 |
| AUROC mask | "体掩码内" | 17/21 脚本全图 | **需重算** |
| K 消融方向 | t≤K₀（数据端） | eval_derive 取最大 K 个（源端） | 已修复 |

## 可复现性检查清单

| 项 | 路径 | 状态 |
|---|---|---|
| 训练配置 | configs/train_ixi_*.yaml | ✅ 存在 |
| 推理配置 | sampler.py 默认参数 | ✅ 记录在案 |
| 数据划分 | data/ixi_slices/manifest_v3.csv | ✅ 存在（含 subject/split） |
| 数据预处理 | prepare_ixi.py / prepare_brats2018.py | ✅ 存在 |
| 评估脚本 | scripts/eval_*.py（12 个） | ✅ 存在 |
| UQ 参数 | definitions_v1.md | ✅ 冻结 |
| 统计方法 | eval_patient_stats.py | ✅ 存在（患者级 bootstrap B=5000） |
| 随机种子 | seed=2026（全部实验统一） | ✅ 记录在案 |
| checkpoint 哈希 | eval_maps_full.py manifest | ✅ 自动记录 |
| 结果文件 | results/*/summary.json + per_slice.csv | ✅ 全部保留 |
| 训练权重 | checkpoints/*.ckpt | ✅ 存在（不公开） |

## 环境记录

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop 8GB |
| Python | 3.10+ |
| PyTorch | 2.x + CUDA |
| 关键依赖 | lightning, scipy, scikit-image, nibabel, pydicom |
| 完整依赖 | requirements_cloud.txt |

## 不可复现项（如实声明）

| 项 | 原因 |
|---|---|
| 官方 IXI 权重的 31.56 dB | 原作者未公开预处理（复现危机 §4.2） |
| SynthRAD T2w→CT 的患者间方差 | 仅 1 名测试患者 |
| 跨机构 C 组的患者间方差 | 每组 6 例，CI 宽 |
| 256² 训练 | 4060 8GB 不可行，AutoDL 包就绪但未启动 |
