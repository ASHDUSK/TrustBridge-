# 数据与权重来源声明（DATA.md）

本仓库遵循"只分发声明的科研数据派生物，不分发原始数据"的原则。

## 1. 数据集

| 数据集 | 用途 | 本仓库中的形态 | 来源与许可 |
|---|---|---|---|
| IXI | 脑 MRI T1/T2/PD 翻译（训练/评估/演示） | `app/examples/` 12 张单切片 PNG（演示输入）；`results/` 中指标数值 | [brain-development.org](https://brain-development.org/ixi-dataset/)，科研免费使用；使用者需自行从官方渠道下载完整数据 |
| BraTS 2018 | 脑肿瘤案例信任评估 | `results/2026-09-28_*` 逐片指标 CSV；论文插图渲染 | [S BraTS 2018 官方渠道](https://www.med.upenn.edu/sbia/brats2018.html)，需签署使用协议；**原始数据不得再分发**，本仓库不含任何原始数据或 NIfTI |
| SynthRAD 2023 | 盆腔 MRI→CT（sCT） | `results/2026-09-27_CT_*` 面板渲染图与逐片指标 | [SynthRAD2023 Challenge](https://synthrad2023.grand-challenge.org/)，科研许可；原始数据不得再分发 |

规则（对贡献者）：

1. 任何 `.nii / .nii.gz / .dcm / .npy / .ckpt` 等原始数据或权重文件**不得提交**（.gitignore 已拦截）；
2. 新增数据派生物（渲染图、指标表）必须在本文件登记来源与许可；
3. 所有数据仅限科研与教学用途，不得用于临床决策。

## 2. 模型权重

| 权重 | 来源 | 分发方式 |
|---|---|---|
| 官方 SelfRDB 权重（ixi_*.ckpt、brats_*.ckpt、ct_*.ckpt） | [icon-lab/SelfRDB](https://github.com/icon-lab/SelfRDB)（MIT） | `scripts/download_checkpoints.sh` 从官方地址下载，不入库 |
| 自训权重（self_*.ckpt） | 本项目基于上述公开数据自训 | 不随仓库分发；获取方式见 `scripts/download_checkpoints.sh` 头注释 |
| 门控系数（trustbridge/modality_gate*.npz） | 本项目自训的轻量逻辑回归/Mahalanobis 分类器系数（无患者数据） | 随代码分发（MIT） |

## 3. 演示切片说明

`app/examples/` 的 12 张 PNG 为 IXI 公开数据集的单张轴向切片导出（8-bit PNG），
仅用于界面演示与功能验证，不代表任何个体患者的完整影像。
