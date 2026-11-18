# TrustBridge 架构说明（docs/architecture.md）

> 本文依据仓库真实源码编写，描述各模块职责、输入输出与调用关系。

## 一、分层结构

```
                        third_party/SelfRDB（vendored 官方代码，MIT）
                        diffusion.DiffusionBridge + backbones.ncsnpp.NCSNpp
                                       ▲ sys.path 注入 + import（engine.py）
                                       │
checkpoints/*.ckpt ──▶ engine.BridgeEngine（12 任务注册表 TASKS）
                                       │ 构造参数传入（self.eng）
                                       ▼
                    sampler.TrustSampler（核心创新层）
                    RCRF + 随机桥集成 + 自适应递归停机 + 步长跳跃
                         │                    ▲
                         ▼                    │ 继承（仅 eval_repaint 使用）
                    pipeline.translate_single / translate_volume    repaint.RepaintSampler
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   app/app.py（Gradio）           api.py（FastAPI /gate /evaluate）
   io_utils · report · metrics    io_utils · report · gate_v2
                                        │
                                        ▼
                                 gate.gate_features（v1 系数 npz）
```

## 二、模块职责

| 模块 | 职责 | 输入 → 输出 | 被谁使用 |
|---|---|---|---|
| `engine.py` | `BridgeEngine`：按任务加载 SelfRDB ckpt，剥离 Lightning，只保留生成器 Gθ 与扩散调度；`TASKS` 注册 12 个任务（IXI 四方向官方/自训、BraTS 两方向、盆腔 CT 官方/自训） | task_id + ckpt 路径 → 引擎对象（含 `.diffusion`） | app、api、全部评估脚本 |
| `sampler.py` | `TrustSampler`：在官方 `sample_x0` 反向采样之上计算 ①CReF 递归收敛残差场 ②N 次桥集成的逐像素标准差 ③自适应递归停机 ④步长跳跃；`TrustResult` 携带 trust_map / rcrf_map / unc_map / trust_score / qc_flag | engine + 输入张量（模型域 [-1,1]）→ `TrustResult` | pipeline、repaint、评估脚本 |
| `pipeline.py` | 推理管线：`get_sampler`（缓存采样器、按任务族设融合权重 0.6/0.4）、`translate_single`（单切片）、`translate_volume`（整卷逐层） | engine + [0,1] 图像数组 → TrustResult + 指标 | app、api、评估脚本 |
| `gate.py` | 输入门控 v1：T1/T2 二分类逻辑回归（`modality_gate.npz`，纯 numpy），体区标准化 + 分位数/梯度特征；导出 `gate_features()` | 图像数组 → 特征向量 / 判别结果 | gate_v2、gate1_verify、eval_gate |
| `gate_v2.py` | 输入门控 v2：6 类模态/解剖分类（brain_T1/T2/FLAIR、pelvis_T1w/T2w/CT）+ 两条 OOD 硬规则（平坦图、Mahalanobis 距离）；`gate_decision()` 输出 MATCH / MISMATCH / OOD 三档策略（脑任务硬拒绝，CT 任务警示放行） | 图像数组 + 任务信息 → 判定与动作 | app、api `/gate`；训练脚本 `train_gate_v2.py` |
| `metrics.py` | `psnr_ssim()` 等指标（skimage 实现） | 两幅 [0,1] 图像 → PSNR/SSIM | pipeline、app、评估脚本 |
| `report.py` | 质控报告：`compose_report_png/pdf/json`、`make_overlay`（低信任红色叠加）、`compose_trust_hist`、`lowtrust_regions`；matplotlib Agg 后端 | TrustResult + 图像 → PNG/PDF/JSON 报告 | app、api `/evaluate` |
| `io_utils.py` | `load_image()`（PNG/JPG/NIfTI/DICOM → [0,1] 数组）、`save_dicom()`（Secondary Capture）、`load_volume()/save_volume()`（NIfTI） | 文件路径 ↔ 数组 | app、api |
| `intensity.py` | Nyúl 式分位数匹配强度后处理（`load_template`/`harmonize01`）；模板文件 `data/intensity_template.npz` 不随仓库分发 | 图像 + 模板 → 校正图像 | eval_brats_trust、sync_demo_package |
| `repaint.py` | `RepaintSampler(TrustSampler)`：RePaint 式区域重采样（重采样点 t* 上 J 条独立子链、按块择优） | engine + 输入 → TrustResult | 仅 eval_repaint（实验性） |

## 三、单次翻译请求的数据流

1. `io_utils.load_image` 读入任意格式 → [0,1] float32 HxW + 元信息；
2. `gate_v2.gate_decision` 提特征（复用 `gate.gate_features`）→ MATCH 放行 / MISMATCH、OOD 拒绝（不合成、无评分）；
3. `engine.BridgeEngine` 持有 Gθ 与 `DiffusionBridge`；
4. `pipeline.get_sampler(engine)` 取缓存 `TrustSampler` → `translate_single` 调 `sample()`（N 次桥采样 + 递归残差）→ `TrustResult`；
5. `metrics.psnr_ssim`（有参考值时）计算 PSNR/SSIM；
6. `report.compose_report_png/pdf/json` 生成四联图（源/译/信任图/低信任叠加）与结构化报告；
7. `io_utils.save_dicom / save_volume` 导出 DICOM Secondary Capture 或 NIfTI。

## 四、脚本入口约定

- `scripts/` 下评估脚本统一以 `ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` 定位仓库根后 `import trustbridge`——因此脚本必须保持在 `scripts/` 一级目录内；
- `scripts/train.py` 包装 `third_party/SelfRDB` 的 LightningCLI（官方 `main.py` 依赖其所在目录作为导入根）；
- 权重统一落在 `checkpoints/`（gitignore），由 `scripts/download_checkpoints.sh` 下载官方权重。
