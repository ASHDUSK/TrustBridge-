# TrustBridge · Rethinking Uncertainty Quantification for Diffusion-Bridge Medical Image Translation

> **TrustBridge 是为医学图像翻译设计的临床安全分诊护栏**（基于 SelfRDB 自洽递归扩散桥，
> Arslan et al., arXiv:2405.06789, MIT）。它在合成**之前**拦截错模态/域外/退化输入
> （1,160 例验证矩阵：562 例非法输入 0 漏放），在合成**之后**给出逐像素信任图与分诊建议
> （PASS→REVIEW→REJECT），并把**自己守卫的每一种失效**（自信的幻觉、基座模型结构性崩塌、
> 退化盲区）作为边界地图公开发表——承认 AI 的局限并设计与之共存的流程，比宣称万能更接近临床可信。

---

## ⚠ Reproducibility Crisis & Our Solution（请先读这段）

官方 SelfRDB 权重在**任何合理方式自备的 IXI 数据**上只能跑到 **~15 dB**
（论文报告 31.56 dB）。我们系统排除了错位 / z 偏移 / 强度窗 / 裁剪 / 方向等
所有可想到的制备差异后，确认差距来自**作者未公开的预处理细节**
（社区在 SynDiff 仓库也报告了同样的 ~17 dB 问题）。

我们的处理方式，也是本仓库对社区的三层价值：

1. **排雷**：完整公开排查过程与自训管线（docs/03），自训同配方模型
   线性强度校正后 **31.74 / 32.52 dB vs 论文 31.56**——内容保真度同级；
2. **揭示评估陷阱**：背景空气像素会系统性虚高整图口径的 UQ 指标
   （同一信号整图 AUROC 0.90 → 体区内 0.78；分离度 35–73× → ~2×）。
   **所有**信任指标必须体掩码口径计算——我们撤回了自己早期的虚高数字；
3. **一键评估你的模型**：`scripts/evaluate_your_model.py` 让你把自己的翻译模型
   接进来，直接得到体掩码口径的 AUROC / ΔPSNR@20% / 十分位分离（见下）。

## 📊 关键结果（体掩码口径 · 等预算基线对照 · 全部可复现）

| 发现 | 数字 |
|---|---|
| T1→T2 自训模型，CReF 单链（**零额外成本**）定位高误差十分位 | AUROC **0.88** |
| 官方权重，真实肿瘤患者（BraTS，外部数据） | AUROC **0.91**，剔除最低信任 20% 体素 PSNR **+3.1 dB** |
| **跨机构迁移**（CBICA 验证集，从未见过的机构） | AUROC **0.95±0.02** |
| 等预算下 CReF vs TTA vs 输入噪声 | **按 (模型, 方向) 而异**——无万能信号 |
| **失效边界**：官方 T2→T1 残差纹理主导 | 全部分歧类信号 ≈ 随机（0.47–0.51） |
| **肿瘤盲区**：信任不定位肿瘤 | AUROC 0.51（瘤内信任反而更高） |
| 输入门控（六类 + Mahalanobis OOD） | 1,160 例矩阵，**562 例非法输入 0 漏放** |
| 崩盘甄别：官方模型 13 dB"崩塌"实为纯强度偏移（NCC≈0.9） | 信任图不报警是**正确语义**（结构确实没坏），但光度失败需另行检查 |

全部阴性结果（口径陷阱、失效边界、肿瘤盲区、图像级 QC 失效、routing 假设否定、
退化响应盲区）与逐片数据开源——**我们发表的是一张边界地图，不是万能药**。

## 🚀 快速开始

```bash
# 1) 一键评估你自己的模型（体掩码口径 + 整图口径对照）
python scripts/evaluate_your_model.py --demo        # 用本仓库自训模型演示
# 或接入你的模型：模块提供 predict_pair(src01)->(pred01, trust_2d)，见脚本头注释

# 2) 启动本地应用（单图/整卷/批量质控/部署校准/压力测试/历史记录）
启动应用.bat        # 或 python app/app.py → http://127.0.0.1:7860

# 3) 微服务模式（HIS/PACS 集成）
uvicorn api:app --port 8000
#   POST /gate      仅输入门控，毫秒级（错模态/域外/退化拦截）
#   POST /evaluate  翻译+信任分诊 JSON（trust_score/qc_flag/suspicious_regions）
#   GET  /docs      交互式 API 文档
# Docker 骨架见 Dockerfile（GPU 直通，权重/数据卷挂载）

# 4) 复现论文全部实验
python scripts/eval_uq_baselines.py --task T1->T2 --ckpt checkpoints/self_t1t2.ckpt
python scripts/eval_tumor_sparsify.py            # BraTS 体区复检
python scripts/eval_crossinst_brats.py           # 跨机构迁移
python scripts/eval_degradation.py               # 退化响应
python scripts/eval_paper_extras.py collapse --n 20   # 崩盘甄别 / routing / 语义代理
```

## 🖥 应用功能（MedBridge Studio）

- **单图翻译**：PNG/JPG/NIfTI/**DICOM** 进，翻译结果 DICOM（Secondary Capture）+
  质控报告 **PDF（可打印归档）**/PNG/JSON 出；示例切片自动匹配任务
- **六类输入门控**：错模态/错部位/域外输入在合成**之前**拦截（脑任务硬拒绝），
  不产生任何伪造评分
- **整卷 NIfTI**：逐层 Trust Score 曲线（低于阈值切片定位）、翻译卷 NIfTI 导出
- **批量 QC**：错配输入记 `GATE_BLOCKED` 不合成，汇总 CSV + 排序图
- **部署校准向导**：用你机构的一批已知真值切片重整定三级阈值（回应"无域校准机制"）
- **诚实 UI**：结果页明示门控判定、阈值未临床标定、图像级分数的局限

## 📁 仓库结构

```
trustbridge/            创新层核心（engine/sampler/pipeline/gate_v2/report/io_utils）
app/app.py              Gradio 应用（中文界面，五个标签页 + 部署校准）
scripts/                数据准备 / 训练 / 全部论文实验脚本（逐片 CSV + summary.json）
configs/                自训练配置（官方 10 步扩散桥配方，128²）
checkpoints/            官方权重（MIT）+ 自训权重
docs/                   18 篇项目文档（含验证矩阵、复现性排查、论文初稿）
docs/latex/             论文 v0.4（main.tex + references.bib，Overleaf 可编译）
results/                全部实验结果（JSON 汇总 + 逐片 CSV）
```

## 📖 论文

`Rethinking Uncertainty Quantification in Diffusion-Bridge Medical Image
Translation: Pitfalls, Baselines, and Boundaries`（docs/latex/main.tex，v0.4）。
引用本工具请同时引用 SelfRDB 原论文与本作品。

## ⚖ 声明

- 方法基座：F. Arslan et al., *Self-Consistent Recursive Diffusion Bridge*,
  arXiv:2405.06789，官方代码 github.com/icon-lab/SelfRDB（MIT）。
- 数据：IXI（brain-development.org）、BraTS2018、SynthRAD2023，仅用于科研。
- **免责声明**：合成结果与信任评分仅供科研与教学演示，不得直接用于临床诊断决策。
