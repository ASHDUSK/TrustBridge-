# v0.8.3 口径统一修正轮 · 完成报告

日期：2026-11-12（凌晨）
触发：G4 收尾时发现 **评估脚本修复承诺与实际不符**（b7709c5 提交说明声称修了
eval_degradation/eval_body_decile，但实际改动未包含它们；已"修复"的
tumor/crossinst 脚本引用了不存在的 `res.rcrf_raw_map` 属性，重算当场崩溃，
2026-11-10/11 的 tumor 与 crossinst 重算目录为空）。

## 一、根因与修复

| # | 问题 | 修复 |
|---|---|---|
| 1 | 信任图融合权重反转：7 个端到端脚本用 sampler 默认 (CReF 0.4/Ens 0.6)，论文口径 0.6/0.4 | 全部改为从 `smp.rcrf_raw_map`/`unc_raw_map` 手动融合（单次 p99 归一化） |
| 2 | `res.rcrf_raw_map` 属性不存在（原始图在 sampler 实例上）→ tumor/crossinst 重算崩溃 | 改为 `get_sampler(eng).rcrf_raw_map` |
| 3 | 切片级 AUROC 未做体区掩码（degradation/crossinst/tumor/collapse/uq 官方行） | 全部 `auroc(tr[mask], err[mask])` |
| 4 | >128 输入（BRATS 240²）trust 对齐用 crop-only → 负索引截断成 232² | 统一 `align_map`（小则中心裁剪/大则双线性放大，同 pipeline） |
| 5 | eval_uq_baselines 对已归一化图二次归一化（double-norm，T2→T1 自训偏差 0.018） | 统一为原始图单次 p99 |
| 6 | derived.json 全部为 K 方向修复前产物（11-09 01:59 < 修复 11-10 23:23），旧方向对拍=0.8051 实锤 | 10 组中 6 组重生成（另 4 组 npz 尺寸不齐，其数字本就走独立脚本） |
| 7 | patient_stats/qc_thresholds/make_figures 硬编码旧日期数据源 | 全部指向 2026-11-12 产物 |

## 二、全量重算（2026-11-12，RTX 4060 本地）

退化×2（自训+官方）、tumor_sparsify、tumor_corrected、crossinst、body_decile、
uq_baselines×4（自训×2+官方×2）、collapse、slice_scores×2、patient_stats、
qc_thresholds、掩码阈值敏感性（新）、权重扫描（新）——全部在统一口径下重跑。
旧 derived.json 备份为 `derived_stale_kdir.json.bak` 留痕。

## 三、头条数字修正（全部为整图口径伪影的撤回/修正）

| 主张 | 旧值（整图口径） | 新值（体区口径） |
|---|---|---|
| 肿瘤队列高误差定位 | 0.911±0.037 | **0.624±0.113**（+3.19dB 稀疏化、2.1× 分离幸存）——**第二次撤回** |
| 跨机构 CBICA | 0.950±0.021 | **0.834±0.045** |
| 留出患者 | 0.863±0.155 | **0.707±0.113** |
| 官方 T2→T1"全盲" | 全部信号系统性低于随机（fused CI [0.442,0.485]） | 本文信号**近随机**（fused 0.567±0.066），TTA 基线保留弱信号 0.630——无人低于随机 |
| 官方 T1→T2 融合 | 0.783 | **0.616±0.041** |
| 崩塌模型"全 PASS 94.2" | 94.2/100 全 PASS | **62.4/100，0 PASS/18 REVIEW/2 REJECT，像素 AUROC 0.61**（健康 82.4/0.82）——叙事从"完全失明"改为"方向正确但远不够用" |
| 表 2 自训行 | 0.823/0.757 | **不变**（0.823/0.757，uq_baselines 重跑复现）；Sep 194→182、4.0→3.7 |

新增诚实发现（写入正文）：
- **融合 vs 仅集成在两方向都显著**（ΔAUROC +0.012 CI[0.003,0.021]；+0.016 CI[0.003,0.029]）
- **SynthRAD 跨解剖**：T1w→CT 0.666±0.061 尚可；T2w→CT 0.591±0.070 近随机；
  K₀ 扫描 0.628→0.743、融合权重排序**反转**（纯集成 0.698 > 纯 CReF 0.575）——
  冻结算子是（模型,方向,解剖）三元组依赖的
- 掩码阈值敏感性（脑区 ±0.013 小；盆腔大）
- 权重扫描：脑区 w∈[0.2,0.6] 宽平台；两端走低

## 四、论文更新（v0.8.3，中英双稿）

- 摘要/贡献段：修正全部头条数字+两处撤回说明（修复了 v0.8.2 摘要漏改与
  "a single single-chain"语法 bug）
- 新增：Tab impl（实现与分析配置参数表）、§4.4 冻结算子敏感性段、
  §4.10 SynthRAD 跨解剖小节
- 重写：§4.2 崩塌段、§4.8 肿瘤段、§4.9 退化（自训/官方配对对照）+跨机构段、
  §4.6 QC 操作点（Pearson +0.49/+0.31/−0.02；PASS≥75 放行 83% 但只捕获 18% 故障）
- 图：fig_body_decile、fig_collapse（标题图例改为"部分分离/操作不足"）、
  fig6_tumor 全部按新数据重生成
- 编译：tectonic main.pdf（18 页）通过；正式初稿走 tex2html+Edge 管线
  （修复 tex2html 对 p{} 列宽表格的解析），TrustBridge_paper_draft.pdf（20 页）/
  _zh.pdf（16 页）生成；注意 **main_zh.pdf 不能用 tectonic 编译**（无 CJK 包，
  会静默丢字形）——中文初稿必须走 paper_pdf.py --zh

## 五、验收

- dry-pass 锚点 41+4 对全中后应用
- visual-judge 子代理通道不可用（账号连接问题），按协议回退自查：
  关键页（摘要/参数表/主表/五张图/改写段）逐页目检通过；0.911 残留均为撤回语境

## 六、遗留

1. app/app.py 与 api.py 仍用 `res.trust_map`（反转权重）——应用端显示口径待统一
2. maps_full 的 tumor_brats/crossinst_B/C/collapse_official 四组 npz 存在
   pred/mask 尺寸不齐（232² 伪影），derived 无法重生成；其论文数字均走独立脚本，不受影响
3. 256² AutoDL 训练、试用反馈收集（用户挂起中）
4. 语义代理边界实验（§4.7-(7)）的 0.42–0.67 来自旧配置运行，结论方向不受影响，
   如需数字级一致可在下轮重跑 eval_paper_extras semantic
