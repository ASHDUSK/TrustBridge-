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

import csv
import json
import shutil

import numpy as np
import glob

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


def list_history(limit=20):
    """扫描 outputs/ 会话目录，返回最近的质量记录（新→旧）。"""
    rows = []
    if not os.path.isdir(OUT_DIR):
        return rows
    for d in sorted(glob.glob(os.path.join(OUT_DIR, "20*")), reverse=True)[:limit * 2]:
        jf = os.path.join(d, "qc_report.json")
        pf = os.path.join(d, "qc_report.png")
        if not os.path.isfile(jf):
            continue
        try:
            j = json.load(open(jf, encoding="utf-8"))
            rows.append({"时间": os.path.basename(d)[:15].replace("_", " ", 1),
                         "任务": j.get("task", "?"),
                         "TrustScore": j.get("trust_score", "?"),
                         "低信任%": j.get("lowtrust_ratio_pct", "–"),
                         "QC": j.get("qc_flag", "?"),
                         "PSNR": (j.get("quality_metrics_official_protocol") or {}).get("psnr", "–"),
                         "目录": d})
        except Exception:
            continue
        if len(rows) >= limit:
            break
    return rows


def clear_history() -> tuple:
    """清空 outputs 下的全部历史质控记录（不可恢复）。"""
    removed = 0
    for d in glob.glob(os.path.join(OUT_DIR, "20*")):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return [], [], "已清空 %d 条历史记录（outputs 空间已释放）。" % removed


PRESETS = {  # (显示名, n_ensemble, step_skip, 说明)
    "快速 ⚡": (1, 2, "单次采样·步跳2，秒级出图"),
    "平衡 ⚖": (4, 2, "4次集成·步跳2（消融实测：质量持平，耗时省40%），推荐"),
    "精细 🔬": (8, 1, "8次集成·逐步采样，最可靠的信任图"),
}


def _session_dir() -> str:
    d = os.path.join(OUT_DIR, datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    os.makedirs(d, exist_ok=True)
    return d


def _score_html(res, low_thresh: float = 0.6) -> str:
    """主视觉 = 低信任区域占比（稳健量）；总分降为参考值并注明限制（limitation ②）。"""
    flag = res.qc_flag
    color = rpt.FLAG_COLOR.get(flag, "#555")
    text = rpt.FLAG_TEXT.get(flag, flag)
    return (
        f"<div style='display:flex;gap:18px;align-items:center;padding:10px 4px;'>"
        f"<div style='font-size:40px;font-weight:800;color:{color};'>"
        f"{res.lowtrust_ratio:.1f}<span style='font-size:18px;color:#888;'>%</span></div>"
        f"<div>"
        f"<div style='font-size:16px;font-weight:700;color:{color};'>"
        f"低信任区域占比（trust &lt; {low_thresh:.2f}）· 质控建议 {flag} · {text}</div>"
        f"<div style='font-size:12px;color:#666;margin-top:2px;'>"
        f"参考总分 Trust Score {res.trust_score:.1f}/100（全图信任均值）</div>"
        f"<div style='font-size:11.5px;color:#888;margin-top:4px;line-height:1.45;'>"
        f"图像级简单聚合与真实误差的相关性有限（IXI +0.37 / CT −0.63，已记入 limitation），"
        f"请以像素级信任图与低信任区域占比为准；"
        f"阈值 PASS≥75 / REVIEW≥55 为演示默认值，未经临床标定。</div>"
        f"</div></div>"
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

    try:
        out = translate_single(eng, img, n_ensemble=n_ens, step_skip=skip,
                               seed=int(seed), gt01=gt01, progress_cb=cb)
    except RuntimeError as e:
        low = str(e).lower()
        if "out of memory" in low or ("cuda" in low and "memory" in low):
            raise gr.Error("显存不足（OOM）：请改用「快速」或「平衡」档位、缩小输入尺寸，"
                           "或关闭其它占用显存的程序后重试。")
        raise gr.Error("推理失败：%s" % str(e)[:200])
    except Exception as e:
        raise gr.Error("处理失败：%s" % str(e)[:200])
    res, pred = out["result"], out["pred01"]

    sdir = _session_dir()
    png_path = os.path.join(sdir, "qc_report.png")
    json_path = os.path.join(sdir, "qc_report.json")
    rpt.compose_report_png(img, pred, res, png_path, gt01=gt01,
                           low_thresh=float(low_thresh),
                           source_mod=src_mod, target_mod=tgt_mod,
                           input_gate=gate_result)
    extra = {"source_modality": src_mod, "target_modality": tgt_mod}
    if gate_result is not None:
        extra["input_gate"] = ("PASS" if gate_result.get("ok", True) else "FLAGGED")
        extra["input_gate_prob_t2"] = gate_result.get("prob_t2")
    rpt.compose_report_json(file.name if hasattr(file, "name") else "input",
                            display, res, json_path, metrics=out["metrics"],
                            extra=extra)

    hist_path = os.path.join(sdir, "trust_hist.png")
    rpt.compose_trust_hist(res, hist_path, low_thresh=float(low_thresh))
    tmap_path = os.path.join(sdir, "trust_map.png")
    rpt.compose_trust_map_png(res.trust_map, tmap_path, low_thresh=float(low_thresh))

    # 区域级证据：把像素级信任图升级成"哪几块需要复核"
    regions = rpt.lowtrust_regions(res.trust_map, float(low_thresh))
    if regions:
        with open(os.path.join(sdir, "lowtrust_regions.json"), "w", encoding="utf-8") as f:
            json.dump({"lowtrust_thresh": float(low_thresh), "regions": regions},
                      f, ensure_ascii=False, indent=2)

    gallery = [
        (img, "源图像"),
        (pred, f"翻译结果 ({tgt_mod})"),
        (tmap_path, "信任图（附色标：亮 = 越可信）"),
        (rpt.make_overlay(pred, res.trust_map, float(low_thresh)), "低信任叠加（红色 = 需复核）"),
        (hist_path, f"信任度分布 · 低信任区域 {res.lowtrust_ratio:.1f}%"),
    ]
    dev = getattr(eng, "device", "?")
    metric_txt = (f"**设备** {dev}  ·  **耗时** {res.elapsed_s:.1f}s  ·  "
                  f"**融合权重** w_rcrf={res.w_used[0]} / w_unc={res.w_used[1]}\n\n")
    metric_txt += "未提供参考真值（演示模式）" if not out["metrics"] else \
        f"PSNR = {out['metrics']['psnr']} dB   ·   SSIM = {out['metrics']['ssim']} %（官方口径）"
    if regions:
        metric_txt += ("\n\n**低信任连通区域 Top-%d（建议重点复核）**\n\n"
                       "| 面积(px) | 占比(%%) | 质心(行,列) | 区域平均信任 | 区域最低信任 |\n"
                       "|---|---|---|---|---|\n" % len(regions))
        for r in regions:
            metric_txt += ("| %d | %.2f | %s | %.3f | %.3f |\n"
                           % (r["面积(px)"], r["占比(%)"], r["质心(行,列)"],
                              r["区域平均信任"], r["区域最低信任"]))
    else:
        metric_txt += f"\n\n未检出显著低信任连通区域（阈值 {float(low_thresh):.2f}）"
    return (gallery, _score_html(res, float(low_thresh)), metric_txt,
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

    # 输入模态门控（取首个含体数据切片校验；CT 任务盆腔门控不可靠，跳过）
    gate_result = None
    gate_note = ""
    if not task_id.startswith("CT_"):
        from trustbridge.gate import check as gate_check, gate_for_task
        body_slice = next((k for k in range(vol.shape[2])
                           if (vol[:, :, k] > 0.02).mean() > 0.05), None)
        if body_slice is not None:
            gate_result = gate_check(vol[:, :, body_slice], src_mod,
                                     gate_path=gate_for_task(task_id))
            if gate_result is not None and not gate_result.get("ok", True):
                gate_note = (f"⚠ **输入模态校验未通过**：该卷特征更接近 "
                             f"{'T2' if gate_result.get('prob_t2', 0) > 0.5 else 'T1'} 加权像，"
                             f"与任务源模态（{src_mod}）不符——结果可能不可靠。")
                gr.Warning(gate_note)

    def cb(f, msg):
        progress(f, desc=msg)

    try:
        outv = translate_volume(eng, vol, n_ensemble=n_ens, step_skip=skip,
                                seed=0, progress_cb=cb)
    except RuntimeError as e:
        low = str(e).lower()
        if "out of memory" in low or ("cuda" in low and "memory" in low):
            raise gr.Error("显存不足（OOM）：整卷模式建议用「快速」档位，或先跑单张切片预览。")
        raise gr.Error("整卷推理失败：%s" % str(e)[:200])
    except Exception as e:
        raise gr.Error("整卷处理失败：%s" % str(e)[:200])
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

    summary = (f"共处理 {len(zs)} 个含体切片 / 全卷 {vol.shape[2]} 层 · 耗时 {time.time()-t0:.0f}s · "
               f"平均 Trust Score = {np.mean(scores):.1f} · 最低 {np.min(scores):.1f} (z={zs[int(np.argmin(scores))]})"
               + (f"  |  {gate_note}" if gate_note else ""))
    return prev_path, score_path, summary, nii_path


IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".nii", ".gz", ".dcm"}

BATCH_FIELDS = ["file", "task", "preset", "input_gate", "trust_score",
                "lowtrust_ratio_pct", "qc_flag", "n_lowtrust_regions",
                "max_region_pct", "psnr_db", "elapsed_s", "w_rcrf", "w_unc",
                "lowtrust_thresh"]


def _collect_inputs(dir_path, files, max_n):
    """批量输入：文件夹路径优先，其次上传的多文件。返回 (路径列表, 提示)。"""
    paths = []
    if dir_path and str(dir_path).strip():
        d = str(dir_path).strip().strip('"')
        if not os.path.isdir(d):
            raise gr.Error("文件夹不存在：%s" % d)
        for root, _dn, fn in os.walk(d):
            for x in sorted(fn):
                if os.path.splitext(x)[1].lower() in IMG_EXT:
                    paths.append(os.path.join(root, x))
        note = "文件夹 %s" % d
    elif files:
        for f in files:
            p = f.name if hasattr(f, "name") else str(f)
            if os.path.splitext(p)[1].lower() in IMG_EXT:
                paths.append(p)
        note = "上传 %d 个文件" % len(files)
    else:
        raise gr.Error("请填写文件夹路径，或上传多个影像文件")
    paths = sorted(paths)
    if not paths:
        raise gr.Error("未找到可处理的影像文件（支持 png/jpg/tif/nii/dcm）")
    if max_n and len(paths) > int(max_n):
        note += f"（共 {len(paths)} 个，按上限取前 {int(max_n)} 个）"
        paths = paths[:int(max_n)]
    else:
        note += f"（共 {len(paths)} 个）"
    return paths, note


def _batch_figure(rows, path, low_thresh):
    """汇总图：按低信任占比排序的条形图 + QC 分级统计。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    order = sorted(rows, key=lambda r: r["lowtrust_ratio_pct"], reverse=True)
    names = [os.path.basename(r["file"])[:22] for r in order]
    vals = [r["lowtrust_ratio_pct"] for r in order]
    cols = [rpt.FLAG_COLOR.get(r["qc_flag"], "#555") for r in order]
    fig, ax = plt.subplots(figsize=(9, max(2.6, 0.34 * len(order) + 1.2)))
    ax.barh(range(len(order)), vals, color=cols)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"低信任区域占比 %（阈值 {low_thresh:.2f}）")
    ax.set_title("批量质控：低信任区域占比排序（红=REJECT 橙=REVIEW 绿=PASS）",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=8, color="#333")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_batch(dir_path, files, task_id, preset, low_thresh, max_n,
              save_each, progress=gr.Progress()):
    """批量质控：逐张翻译 + 汇总 CSV（列与 scripts/eval_gt_trust.py 的口径对齐）。"""
    paths, note = _collect_inputs(dir_path, files, max_n)
    n_ens, skip, _ = PRESETS[preset]
    eng = get_engine(task_id)
    display, src_mod, tgt_mod, _ckpt = TASKS[task_id]

    sdir = _session_dir()
    rows, worst, flagged = [], [], []
    t0 = time.time()
    for i, p in enumerate(paths):
        progress(float(i) / max(len(paths), 1), desc=f"({i+1}/{len(paths)}) {os.path.basename(p)}")
        try:
            img, _meta = load_image(p)
            # 输入模态门控：批量里尤其有用——能直接揪出"喂错序列"的病例
            gate_txt = "skipped"
            if not task_id.startswith("CT_"):
                try:
                    from trustbridge.gate import check as gate_check, gate_for_task
                    g = gate_check(img, src_mod, gate_path=gate_for_task(task_id))
                    gate_txt = "PASS" if (g is None or g.get("ok", True)) else "FLAGGED"
                    if gate_txt == "FLAGGED":
                        flagged.append(os.path.basename(p))
                except Exception:
                    gate_txt = "n/a"
            out = translate_single(eng, img, n_ensemble=n_ens, step_skip=skip, seed=0)
            res = out["result"]
            regions = rpt.lowtrust_regions(res.trust_map, float(low_thresh))
            row = {
                "file": os.path.basename(p),
                "task": display,
                "preset": preset,
                "input_gate": gate_txt,
                "trust_score": round(float(res.trust_score), 2),
                "lowtrust_ratio_pct": round(float(res.lowtrust_ratio), 2),
                "qc_flag": res.qc_flag,
                "n_lowtrust_regions": len(regions),
                "max_region_pct": round(max([r["占比(%)"] for r in regions], default=0.0), 2),
                "psnr_db": "" if not out["metrics"] else round(out["metrics"]["psnr"], 2),
                "elapsed_s": round(float(res.elapsed_s), 1),
                "w_rcrf": res.w_used[0],
                "w_unc": res.w_used[1],
                "lowtrust_thresh": float(low_thresh),
            }
            rows.append(row)
            worst.append((row["lowtrust_ratio_pct"], p, res, out["pred01"], img))
            if save_each:
                sub = os.path.join(sdir, "per_case", "%03d_%s" % (i + 1, row["file"]))
                os.makedirs(sub, exist_ok=True)
                try:
                    rpt.compose_report_png(img, out["pred01"], res,
                                           os.path.join(sub, "qc_report.png"),
                                           low_thresh=float(low_thresh),
                                           source_mod=src_mod, target_mod=tgt_mod)
                    rpt.compose_report_json(row["file"], display, res,
                                            os.path.join(sub, "qc_report.json"),
                                            metrics=out["metrics"])
                except Exception:
                    pass
        except Exception as e:
            rows.append({"file": os.path.basename(p), "task": display, "preset": preset,
                         "input_gate": "n/a",
                         "trust_score": "", "lowtrust_ratio_pct": "", "qc_flag": "ERROR",
                         "n_lowtrust_regions": "", "max_region_pct": "",
                         "psnr_db": "", "elapsed_s": "", "w_rcrf": "", "w_unc": "",
                         "lowtrust_thresh": float(low_thresh)})
            gr.Warning(f"跳过 {os.path.basename(p)}：{str(e)[:80]}")

    csv_path = os.path.join(sdir, "batch_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=BATCH_FIELDS)
        w.writeheader()
        w.writerows(rows)
    fig_path = _batch_figure([r for r in rows if r["qc_flag"] != "ERROR"],
                             os.path.join(sdir, "batch_overview.png"), float(low_thresh))

    ok = [r for r in rows if r["qc_flag"] != "ERROR"]
    n_pass = sum(1 for r in ok if r["qc_flag"] == "PASS")
    n_rev = sum(1 for r in ok if r["qc_flag"] == "REVIEW")
    n_rej = sum(1 for r in ok if r["qc_flag"] == "REJECT")
    lt = [r["lowtrust_ratio_pct"] for r in ok]
    md = (f"**批量质控完成**：{note} · 成功 {len(ok)}/{len(rows)} · "
          f"总耗时 {time.time()-t0:.0f}s · 档位「{preset}」\n\n")
    md += (f"| PASS | REVIEW | REJECT | 平均低信任% | 最高 |\n|---|---|---|---|---|\n"
           f"| {n_pass} | {n_rev} | {n_rej} | "
           f"{np.mean(lt) if lt else 0:.1f} | {max(lt) if lt else 0:.1f} |\n")
    worst.sort(key=lambda x: x[0], reverse=True)
    if worst:
        md += "\n**最需要复核的 3 例**（低信任占比最高）\n\n| 文件 | 低信任% | 质控 | 区域数 |\n|---|---|---|---|\n"
        for v, p, res, _pred, _img in worst[:3]:
            md += (f"| {os.path.basename(p)} | {v:.1f} | {res.qc_flag} | "
                   f"{len(rpt.lowtrust_regions(res.trust_map, float(low_thresh)))} |\n")
    if flagged:
        md += ("\n\n⚠ **输入模态校验未通过 %d 例**（特征与任务源模态 %s 不符，"
               "结果不可靠，请核对序列）：%s"
               % (len(flagged), src_mod, "、".join(flagged[:6])
                  + ("…" if len(flagged) > 6 else "")))
    md += (f"\n汇总 CSV 可直接用于统计/论文表格；"
           f"{'每例报告已存于本次会话目录 per_case/。' if save_each else '（未勾选逐例存档，仅保存汇总）'}")

    gallery = []
    for v, p, res, pred, img in worst[:3]:
        tp = os.path.join(sdir, "worst_%s_trust.png" % os.path.splitext(os.path.basename(p))[0])
        try:
            rpt.compose_trust_map_png(res.trust_map, tp, float(low_thresh))
            gallery.append((tp, f"{os.path.basename(p)} · 低信任 {v:.1f}% · {res.qc_flag}"))
        except Exception:
            pass
    return md, csv_path, fig_path, gallery


ABOUT = f"""
## TrustBridge · 可信医学影像翻译工作台

**方法基座**：SelfRDB —— Self-Consistent Recursive Diffusion Bridge（Arslan 等，arXiv:2405.06789，MIT 协议）。
本项目使用其官方预训练权重（IXI 数据集训练），并加入原创创新层：

| 创新点 | 说明 |
|---|---|
| **① 递归收敛残差场 (RCRF)** | 捕获反向采样最后若干步内、相邻递归估计的逐像素残差：网络"反复修改仍无法自洽"的区域即幻觉候选区，零额外成本 |
| **② 随机桥集成不确定性** | N 次独立桥采样（不同噪声种子/生成器 latent）取逐像素标准差，刻画输出稳定性 |
| **③ 信任图与质控报告** | ①②融合为逐像素信任图 → 低信任区域占比 + 可信度分布 → PASS / REVIEW / REJECT 质控建议 + 可归档报告 |
| **④ 步长跳跃推理加速（skip2）** | 隔步采集信任信号，前向次数 40→24：**耗时省约 39%，定位质量持平**（消融实测）。原块级自适应递归方案经 60 片独立复核无增益（−0.05 dB、耗时 2.7 倍），已归档，不作为创新点 |

**典型场景**：MRI 缺失序列补全（T1↔T2/PD）、MRI→sCT（放疗参考）、AI 合成结果质控归档。

**引用**：如使用本工具请引用 SelfRDB 原论文与本作品。

**免责声明**：合成结果仅供科研与教学演示，不得直接用于临床诊断决策。
"""


TASK_GROUPS = {
    "脑 IXI（T1/T2/PD）": ["T1->T2", "T2->T1", "PD->T1", "T1->PD", "T2->T1*", "T1->T2*"],
    "脑肿瘤 BRATS（T2/FLAIR）": ["BRATS_T2->T1", "BRATS_FLAIR->T2"],
    "盆腔 CT（SynthRAD）": ["CT_T1->CT", "CT_T2->CT", "CT_T1->CT_S", "CT_T2->CT_S"],
}


def _task_picker(tasks_ok, default_task, label="翻译任务"):
    """按数据集族分组的任务选择器：[数据集下拉] + [组内任务单选]。"""
    g_ok = {g: [t for t in ts if t in tasks_ok] for g, ts in TASK_GROUPS.items()}
    g_ok = {g: ts for g, ts in g_ok.items() if ts}
    if not g_ok:
        return None, gr.Dropdown(tasks_ok, value=tasks_ok[0], label=label)
    dflt_g = next((g for g, ts in g_ok.items() if default_task in ts), list(g_ok)[0])
    dflt_t = default_task if default_task in g_ok[dflt_g] else g_ok[dflt_g][0]
    dd = gr.Dropdown(list(g_ok), value=dflt_g, label="数据集")
    radio = gr.Radio(g_ok[dflt_g], value=dflt_t, label=label)

    def _on_group(g):
        return gr.Radio(g_ok[g], value=g_ok[g][0] if g_ok[g] else None)

    dd.change(_on_group, inputs=dd, outputs=radio)
    return dd, radio


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
                    _, task = _task_picker(tasks_ok, default_task)
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
                    vtasks_ok = available_tasks()
                    _, vtask = _task_picker(vtasks_ok,
                                            "T2->T1" if "T2->T1" in vtasks_ok else vtasks_ok[0])
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

        with gr.Tab("📦 批量质控（文件夹 / 多文件）"):
            gr.Markdown(
                "对一个文件夹（或一次上传的多个文件）逐张翻译，输出**汇总 CSV** 与"
                "**低信任占比排序图**，用于快速筛出需要人工复核的病例。"
                "CSV 列口径与 `scripts/eval_gt_trust.py` 对齐。")
            with gr.Row():
                with gr.Column(scale=5):
                    bdir = gr.Textbox(label="影像文件夹路径（本机路径，如 D:\\data\\slices）",
                                      placeholder=r"D:\MedTranslateProject\app\examples")
                    bfiles = gr.File(label="或直接上传多个文件（可不填路径）",
                                     file_count="multiple",
                                     file_types=["image", ".nii", ".gz", ".dcm"])
                    _, btask = _task_picker(available_tasks(),
                                            "T2->T1" if "T2->T1" in available_tasks()
                                            else available_tasks()[0])
                    bpreset = gr.Radio(list(PRESETS.keys()), value="快速 ⚡",
                                       label="质量档位（批量建议快速）")
                    with gr.Row():
                        blow = gr.Slider(0.3, 0.9, value=0.6, step=0.05,
                                         label="低信任标红阈值")
                        bmax = gr.Number(value=20, precision=0,
                                         label="最多处理张数（0=不限）")
                    bsave = gr.Checkbox(value=False,
                                        label="逐例保存质控报告 PNG/JSON（更慢）")
                    bbtn = gr.Button("🚀 批量开始", variant="primary", size="lg")
                with gr.Column(scale=6):
                    bsum = gr.Markdown()
                    bfig = gr.Image(label="低信任占比排序", height=340)
                    bgallery = gr.Gallery(label="最需要复核的 3 例 · 信任图",
                                          columns=3, height="240")
                    bcsv = gr.File(label="汇总 CSV 下载")
            bbtn.click(run_batch, inputs=[bdir, bfiles, btask, bpreset, blow, bmax, bsave],
                       outputs=[bsum, bcsv, bfig, bgallery])

        with gr.Tab("🕓 历史记录"):
            with gr.Row():
                hist_btn = gr.Button("🔄 刷新历史", size="sm")
                clear_btn = gr.Button("🗑 清空历史（不可恢复）", size="sm")
            hist_msg = gr.Markdown()
            hist_df = gr.Dataframe(headers=["时间", "任务", "TrustScore", "低信任%", "QC", "PSNR", "目录"],
                                   interactive=False, wrap=True)
            hist_gallery = gr.Gallery(label="最近的质控报告", columns=4, height="300",
                                      object_fit="contain")

            def refresh_history():
                rows = list_history()
                pics = [(os.path.join(r["目录"], "qc_report.png"), f"{r['任务']} · {r['TrustScore']}")
                        for r in rows
                        if os.path.isfile(os.path.join(r["目录"], "qc_report.png"))][:12]
                return rows, pics

            hist_btn.click(refresh_history, outputs=[hist_df, hist_gallery])
            clear_btn.click(clear_history, outputs=[hist_df, hist_gallery, hist_msg])
            demo.load(refresh_history, outputs=[hist_df, hist_gallery])

        with gr.Tab("📖 关于 / 创新点"):
            gr.Markdown(ABOUT)
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)  # 单 GPU 串行
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, inbrowser=True)
