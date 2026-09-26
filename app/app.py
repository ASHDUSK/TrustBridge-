# -*- coding: utf-8 -*-
"""MedBridge Studio · 医学影像可信翻译工作台

基于 SelfRDB (arXiv:2405.06789, MIT) 官方预训练权重，
TrustBridge 创新层：递归收敛残差场 + 随机桥集成不确定性 + 质控报告。

启动: python app/app.py  （浏览器访问 http://127.0.0.1:7860）
"""
import os
import sys
import time
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import gradio as gr

from trustbridge.engine import BridgeEngine, TASKS
from trustbridge.pipeline import translate_single, translate_volume
from trustbridge import report as rpt
from trustbridge.io_utils import load_image, load_volume, save_volume
from trustbridge.metrics import psnr_ssim

OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

ENGINES: dict[str, BridgeEngine] = {}


def available_tasks() -> list[str]:
    """只列出权重文件实际存在的任务（支持精简分发包）。"""
    ok = []
    for tid, (_disp, _s, _t, ckpt) in TASKS.items():
        if os.path.exists(os.path.join(ROOT, "checkpoints", ckpt)):
            ok.append(tid)
    return ok or list(TASKS.keys())


def get_engine(task_id: str) -> BridgeEngine:
    if task_id not in ENGINES:
        ENGINES[task_id] = BridgeEngine(task_id)
    return ENGINES[task_id]


PRESETS = {  # (显示名, n_ensemble, step_skip, 说明)
    "快速 ⚡": (1, 2, "单次采样·步跳2，秒级出图"),
    "平衡 ⚖": (4, 2, "4次集成·步跳2（消融实测：质量持平，耗时省40%），推荐"),
    "精细 🔬": (8, 1, "8次集成·逐步采样，最可靠的信任图"),
}


def _session_dir() -> str:
    d = os.path.join(OUT_DIR, datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    os.makedirs(d, exist_ok=True)
    return d


def _score_html(score: float, flag: str) -> str:
    color = rpt.FLAG_COLOR.get(flag, "#555")
    text = rpt.FLAG_TEXT.get(flag, flag)
    return (
        f"<div style='display:flex;gap:18px;align-items:center;padding:10px 4px;'>"
        f"<div style='font-size:40px;font-weight:800;color:{color};'>"
        f"{score:.1f}<span style='font-size:18px;color:#888;'> /100</span></div>"
        f"<div><div style='font-size:16px;font-weight:700;color:{color};'>{flag} · {text}</div>"
        f"<div style='font-size:12px;color:#666;'>Trust Score = 融合信任图均值，"
        f"综合递归收敛残差与集成不确定性</div></div></div>"
    )


def run_single(file, task_id, preset, low_thresh, gt_file, seed, slice_k, progress=gr.Progress()):
    if file is None:
        raise gr.Error("请先上传影像文件")
    n_ens, skip, _ = PRESETS[preset]
    eng = get_engine(task_id)
    display, src_mod, tgt_mod, _ckpt = TASKS[task_id]

    img, meta = load_image(file.name if hasattr(file, "name") else str(file),
                           slice_idx=slice_k if slice_k is not None else None)
    gt01 = None
    if gt_file is not None:
        gtp = gt_file.name if hasattr(gt_file, "name") else str(gt_file)
        if os.path.splitext(gtp)[1].lower() in {".nii", ".gz", ".dcm"}:
            gt01, _ = load_image(gtp, slice_idx=meta.get("slice"))
        else:
            gt01, _ = load_image(gtp)
        if gt01.shape != img.shape:
            gt01 = None

    def cb(f, msg):
        progress(f, desc=msg)

    # 输入模态门控（脑任务启用；CT 任务盆腔门控不可靠，跳过——见 docs/08 附录 C）
    gate_result = None
    if not task_id.startswith("CT_"):
        from trustbridge.gate import check as gate_check, gate_for_task
        gate_result = gate_check(img, src_mod, gate_path=gate_for_task(task_id))
        if gate_result is not None and not gate_result.get("ok", True):
            gr.Warning(f"⚠ 输入模态校验未通过：该图像特征更接近 "
                       f"{'T2' if gate_result.get('prob_t2', 0) > 0.5 else 'T1'} 加权像，"
                       f"与所选任务的源模态（{src_mod}）不符。结果可能不可靠，请核对任务选择。")

    out = translate_single(eng, img, n_ensemble=n_ens, step_skip=skip,
                           seed=int(seed), gt01=gt01, progress_cb=cb)
    res, pred = out["result"], out["pred01"]

    sdir = _session_dir()
    png_path = os.path.join(sdir, "qc_report.png")
    json_path = os.path.join(sdir, "qc_report.json")
    rpt.compose_report_png(img, pred, res, png_path, gt01=gt01,
                           low_thresh=float(low_thresh),
                           source_mod=src_mod, target_mod=tgt_mod)
    extra = {"source_modality": src_mod, "target_modality": tgt_mod}
    if gate_result is not None:
        extra["input_gate"] = ("PASS" if gate_result.get("ok", True) else "FLAGGED")
        extra["input_gate_prob_t2"] = gate_result.get("prob_t2")
    rpt.compose_report_json(file.name if hasattr(file, "name") else "input",
                            display, res, json_path, metrics=out["metrics"],
                            extra=extra)

    gallery = [
        (img, "源图像"),
        (pred, f"翻译结果 ({tgt_mod})"),
        (res.trust_map, "信任图"),
        (rpt.make_overlay(pred, res.trust_map, float(low_thresh)), "低信任叠加"),
    ]
    metric_txt = "未提供参考真值（演示模式）" if not out["metrics"] else \
        f"PSNR = {out['metrics']['psnr']} dB   ·   SSIM = {out['metrics']['ssim']} %（官方口径）"
    return (gallery, _score_html(res.trust_score, res.qc_flag), metric_txt,
            png_path, json_path)


def run_volume(file, task_id, preset, progress=gr.Progress()):
    if file is None:
        raise gr.Error("请先上传 NIfTI 卷文件")
    path = file.name if hasattr(file, "name") else str(file)
    if not path.endswith((".nii", ".gz")):
        raise gr.Error("整卷模式请上传 .nii / .nii.gz 文件")
    n_ens, skip, _ = PRESETS[preset]
    eng = get_engine(task_id)
    display, src_mod, tgt_mod, _ckpt = TASKS[task_id]

    vol, ref = load_volume(path)
    t0 = time.time()

    def cb(f, msg):
        progress(f, desc=msg)

    outv = translate_volume(eng, vol, n_ensemble=n_ens, step_skip=skip,
                            seed=0, progress_cb=cb)
    sdir = _session_dir()
    nii_path = os.path.join(sdir, f"translated_{tgt_mod}.nii.gz")
    save_volume(outv["out"], ref, nii_path)

    # 摘要图：中央3层对比 + 每层信任分曲线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    zs = [z for z, _ in outv["slices"]]
    scores = [s for _, s in outv["slices"]]
    show = zs[len(zs) // 2 - 1: len(zs) // 2 + 2] if len(zs) >= 3 else zs
    fig, axes = plt.subplots(2, max(len(show), 1), figsize=(4 * max(len(show), 1), 8.6))
    axes = np.atleast_2d(axes)
    for j, z in enumerate(show):
        axes[0, j].imshow(vol[:, :, z], cmap="gray"); axes[0, j].set_title(f"源 z={z}"); axes[0, j].axis("off")
        axes[1, j].imshow(outv["out"][:, :, z], cmap="gray"); axes[1, j].set_title(f"合成 {tgt_mod} z={z}"); axes[1, j].axis("off")
    fig.suptitle("整卷翻译预览（中央切片）", fontsize=15, fontweight="bold")
    fig.tight_layout()
    prev_path = os.path.join(sdir, "volume_preview.png")
    fig.savefig(prev_path, dpi=130); plt.close(fig)

    fig2, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(zs, scores, marker="o", ms=3, color="#2563eb")
    ax.set_xlabel("轴位切片 z"); ax.set_ylabel("Trust Score")
    ax.set_ylim(0, 100); ax.grid(alpha=0.3)
    ax.set_title("逐层 Trust Score（低于 55 建议人工复核）")
    fig2.tight_layout()
    score_path = os.path.join(sdir, "volume_trust_scores.png")
    fig2.savefig(score_path, dpi=130); plt.close(fig2)

    summary = (f"共处理 {len(zs)} 个含脑切片 / 全卷 {vol.shape[2]} 层 · 耗时 {time.time()-t0:.0f}s · "
               f"平均 Trust Score = {np.mean(scores):.1f} · 最低 {np.min(scores):.1f} (z={zs[int(np.argmin(scores))]})")
    return prev_path, score_path, summary, nii_path


ABOUT = f"""
## TrustBridge · 可信医学影像翻译工作台（大创赛作品）

**方法基座**：SelfRDB —— Self-Consistent Recursive Diffusion Bridge（Arslan 等，arXiv:2405.06789，MIT 协议）。
本项目使用其官方预训练权重（IXI 数据集训练），并加入原创创新层：

| 创新点 | 说明 |
|---|---|
| **① 递归收敛残差场 (RCRF)** | 捕获反向采样最后若干步内、相邻递归估计的逐像素残差：网络"反复修改仍无法自洽"的区域即幻觉候选区，零额外成本 |
| **② 随机桥集成不确定性** | N 次独立桥采样（不同噪声种子/生成器 latent）取逐像素标准差，刻画输出稳定性 |
| **③ Trust Score 与质控门控** | ①②融合为信任图 → 图像级 0-100 分 → PASS / REVIEW / REJECT 三级建议 + 可归档报告 |
| **④ 自适应递归与步跳加速** | 逐图自适应递归停机 + 后验步跳跃，三档质量预设平衡速度与可信度 |

**典型场景**：MRI 缺失序列补全（T1↔T2/PD）、MRI→sCT（放疗参考）、AI 合成结果质控归档。

**引用**：如使用本工具请引用 SelfRDB 原论文与本作品。

**免责声明**：合成结果仅供科研与教学演示，不得直接用于临床诊断决策。
"""


def build_ui():
    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
    with gr.Blocks(title="MedBridge Studio · 医学影像可信翻译", theme=theme) as demo:
        gr.Markdown(
            "<h1 style='margin-bottom:0;'>🩻 MedBridge Studio</h1>"
            "<p style='font-size:15px;color:#555;margin-top:4px;'>"
            "医学影像<b>可信翻译</b>工作台 —— 不只合成目标模态，还告诉你<b>哪里可信</b>"
            "（基座：SelfRDB 扩散桥 · 创新层：TrustBridge）</p>")

        with gr.Tab("🖼 单图翻译"):
            with gr.Row():
                with gr.Column(scale=5):
                    inp = gr.File(label="上传影像（PNG/JPG 切片、.nii/.nii.gz、DICOM）",
                                  file_types=["image", ".nii", ".gz", ".dcm"])
                    gt = gr.File(label="参考真值（可选，用于计算 PSNR/SSIM）",
                                 file_types=["image", ".nii", ".gz", ".dcm"])
                    tasks_ok = available_tasks()
                    default_task = "T2->T1" if "T2->T1" in tasks_ok else tasks_ok[0]
                    task = gr.Radio(tasks_ok, value=default_task,
                                    label="翻译任务")
                    preset = gr.Radio(list(PRESETS.keys()), value="平衡 ⚖",
                                      label="质量档位")
                    with gr.Row():
                        low = gr.Slider(0.3, 0.9, value=0.6, step=0.05,
                                        label="低信任标红阈值")
                        seed = gr.Number(value=0, precision=0, label="随机种子")
                        slice_k = gr.Number(value=None, precision=0,
                                            label="NIfTI 切片号（默认中间层）")
                    btn = gr.Button("🚀 开始翻译", variant="primary", size="lg")
                    ex_dir = os.path.join(ROOT, "app", "examples")
                    if os.path.isdir(ex_dir) and os.listdir(ex_dir):
                        gr.Examples(
                            examples=[os.path.join(ex_dir, f) for f in
                                      sorted(os.listdir(ex_dir)) if f.endswith(".png")][:6],
                            inputs=[inp],
                            label="示例切片（点击加载）")
                with gr.Column(scale=6):
                    score = gr.HTML()
                    metric = gr.Markdown()
                    gallery = gr.Gallery(label="结果（源 / 翻译 / 信任图 / 低信任叠加）",
                                         columns=2, height="460")
                    with gr.Row():
                        f1 = gr.File(label="质控报告 PNG")
                        f2 = gr.File(label="质控报告 JSON")
            btn.click(run_single,
                      inputs=[inp, task, preset, low, gt, seed, slice_k],
                      outputs=[gallery, score, metric, f1, f2])

        with gr.Tab("🧠 整卷翻译 (NIfTI)"):
            with gr.Row():
                with gr.Column():
                    vinp = gr.File(label="上传整卷 .nii / .nii.gz",
                                   file_types=[".nii", ".gz"])
                    vtask = gr.Radio(available_tasks(), value="T2->T1" if "T2->T1" in available_tasks() else available_tasks()[0], label="翻译任务")
                    vpreset = gr.Radio(list(PRESETS.keys()), value="快速 ⚡",
                                       label="质量档位（整卷建议快速）")
                    vbtn = gr.Button("🚀 翻译整卷", variant="primary", size="lg")
                with gr.Column():
                    vsum = gr.Markdown()
                    vprev = gr.Image(label="中央切片预览", height=380)
                    vscore = gr.Image(label="逐层 Trust Score", height=260)
                    vnii = gr.File(label="翻译后 NIfTI 下载")
            vbtn.click(run_volume, inputs=[vinp, vtask, vpreset],
                       outputs=[vprev, vscore, vsum, vnii])

        with gr.Tab("📖 关于 / 创新点"):
            gr.Markdown(ABOUT)
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)  # 单 GPU 串行
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, inbrowser=True)
