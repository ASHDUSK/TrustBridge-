"""质控报告生成：合成图（PNG）+ 结构化报告（JSON）。"""
from __future__ import annotations

import json
import os
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体（Windows 自带）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FLAG_COLOR = {"PASS": "#16a34a", "REVIEW": "#d97706", "REJECT": "#dc2626"}
FLAG_TEXT = {"PASS": "通过", "REVIEW": "建议人工复核", "REJECT": "不建议使用"}


def make_overlay(pred01: np.ndarray, trust_map: np.ndarray,
                 thresh: float = 0.6) -> np.ndarray:
    """翻译图叠加低信任区域（红色半透明）。"""
    rgb = np.repeat(pred01[..., None], 3, axis=2)
    low = trust_map < thresh
    tint = np.array([1.0, 0.15, 0.15])
    a = 0.55
    rgb[low] = (1 - a) * rgb[low] + a * tint
    return np.clip(rgb, 0, 1)


def compose_trust_hist(result, path, low_thresh: float = 0.6):
    """信任图分布直方图：UI 主视觉依据（比单一均值稳健，见 limitation ②）。"""
    hist = list(getattr(result, "trust_hist", []) or [])
    if len(hist) != 10:
        hist = [0.0] * 10
    labels = ["%.1f–%.1f" % (i / 10, (i + 1) / 10) for i in range(10)]
    low_n = int(round(float(low_thresh) * 10))
    colors = ["#dc2626" if i < low_n else "#16a34a" for i in range(10)]

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(range(10), hist, color=colors, edgecolor="#0f172a", linewidth=0.4)
    ax.axvline(low_n - 0.5, color="#111827", ls="--", lw=1.2)
    ax.set_xticks(range(10))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("信任度分箱（0 = 最不可信，1 = 最可信）", fontsize=10)
    ax.set_ylabel("像素占比（%）", fontsize=10)
    ax.set_title("信任度分布  ·  低信任（<%.1f）区域占比 %.1f%%   ·  参考总分 %.1f/100"
                 % (low_thresh, getattr(result, "lowtrust_ratio", 0.0),
                    getattr(result, "trust_score", 0.0)),
                 fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def compose_trust_map_png(trust_map: np.ndarray, path, low_thresh: float = 0.6):
    """带色标的信任图（Gradio gallery 无 colorbar，颜色无法解读 → 单独成图）。"""
    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    im = ax.imshow(trust_map, cmap="viridis", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title("信任图（亮 = 越可信）", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("信任度：1 = 可信，0 = 不可信", fontsize=9)
    cbar.ax.axhline(float(low_thresh), color="#dc2626", lw=1.4)
    cbar.ax.text(1.35, float(low_thresh), " 复核线 %.2f" % low_thresh,
                 color="#dc2626", fontsize=8, va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def lowtrust_regions(trust_map: np.ndarray, low_thresh: float = 0.6,
                     top_k: int = 5, min_area: int = 16) -> list:
    """低信任连通区域 top-k：把像素级证据升级为可人工复核的结构化区域。

    动机：图像级单一均值与真实误差相关性有限（limitation ②），区域级证据更稳健，
    也更接近"指出哪几块需要复核"的临床用法。
    """
    try:
        from skimage import measure
    except Exception:
        return []
    mask = trust_map < float(low_thresh)
    if mask.sum() == 0:
        return []
    lab = measure.label(mask, connectivity=2)
    total = float(trust_map.size)
    out = []
    for p in measure.regionprops(lab):
        if p.area < min_area:
            continue
        cy, cx = p.centroid
        vals = trust_map[lab == p.label]
        out.append({
            "面积(px)": int(p.area),
            "占比(%)": round(p.area / total * 100.0, 2),
            "质心(行,列)": "%d,%d" % (int(cy), int(cx)),
            "区域平均信任": round(float(vals.mean()), 3),
            "区域最低信任": round(float(vals.min()), 3),
        })
    out.sort(key=lambda d: -d["面积(px)"])
    return out[:top_k]


def compose_report_png(source01, pred01, result, path, gt01=None,
                       low_thresh: float = 0.6,
                       source_mod: str = "源", target_mod: str = "目标",
                       input_gate: dict | None = None):
    """2×3 报告图：源 | 翻译 | 信任热图 | 叠加 | 误差(可选) | 收敛曲线。"""
    n_err = gt01 is not None
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2))
    title = "TrustBridge 可信翻译质控报告"
    if input_gate is not None:
        # 注：不要用 ✓ / ⚠ 等符号——微软雅黑缺这些字形，会渲染成方块
        if input_gate.get("ok", True):
            title += "   ｜ 输入校验：通过"
        else:
            title += "   ｜ 输入校验：未通过（模态不符，结果不可靠）"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.985,
                 color="#b91c1c" if input_gate is not None and not input_gate.get("ok", True) else "#0f172a")

    panels = [
        (axes[0, 0], source01, f"源图像（{source_mod}）", "gray", None),
        (axes[0, 1], pred01, f"翻译结果（{target_mod}）", "gray", None),
        (axes[0, 2], result.trust_map, "信任图（亮=可信）", "inferno", "viridis"),
    ]
    for ax, img, title, cmap, _ in panels:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    axes[1, 0].imshow(make_overlay(pred01, result.trust_map, low_thresh))
    axes[1, 0].set_title(f"低信任叠加（trust < {low_thresh} 标红）", fontsize=12)
    axes[1, 0].axis("off")

    if n_err:
        err = np.abs(gt01 - pred01)
        im = axes[1, 1].imshow(err, cmap="hot", vmin=0, vmax=max(err.max(), 1e-6))
        axes[1, 1].set_title("真实误差 |pred−GT|（仅评估用）", fontsize=12)
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
        axes[1, 1].axis("off")
    else:
        axes[1, 1].imshow(result.unc_map, cmap="magma", vmin=0, vmax=1)
        axes[1, 1].set_title("集成不确定性图", fontsize=12)
        axes[1, 1].axis("off")

    # 收敛曲线（首个集成批次）
    curve = result.step_residuals[0] if result.step_residuals else []
    axes[1, 2].plot(range(1, len(curve) + 1), curve, marker="o", color="#2563eb")
    axes[1, 2].set_xlabel("反向扩散步（t 从 T→1）", fontsize=11)
    axes[1, 2].set_ylabel("末次递归残差（L1）", fontsize=11)
    axes[1, 2].set_title("递归收敛曲线", fontsize=12)
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].spines["top"].set_visible(False)
    axes[1, 2].spines["right"].set_visible(False)

    flag_c = FLAG_COLOR.get(result.qc_flag, "#555")
    fig.text(0.5, 0.012,
             f"Trust Score = {result.trust_score:.1f}/100（参考值）   ·   "
             f"低信任区域占比 {result.lowtrust_ratio:.1f}%   ·   "
             f"质控建议: {result.qc_flag}（{FLAG_TEXT.get(result.qc_flag, '')}）   ·   "
             f"集成 N={result.n_ensemble} · 步跳={result.step_skip} · 耗时 {result.elapsed_s:.1f}s",
             ha="center", fontsize=11, color=flag_c, fontweight="bold")
    fig.tight_layout(rect=[0, 0.035, 1, 0.965])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compose_report_json(source_name, task_id, result, path, metrics=None,
                        extra=None):
    payload = {
        "tool": "TrustBridge (MedBridge Studio)",
        "base_model": "SelfRDB (arXiv:2405.06789, MIT License)",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source_file": os.path.basename(str(source_name)),
        "task": task_id,
        "trust_score": round(result.trust_score, 2),
        "trust_score_meaning": "全图信任均值（参考值）；图像级聚合与真实误差相关有限，请以像素级信任图与 lowtrust_ratio_pct 为准",
        "qc_flag": result.qc_flag,
        "qc_note": getattr(result, "qc_note", ""),
        "lowtrust_ratio_pct": round(getattr(result, "lowtrust_ratio", 0.0), 2),
        "lowtrust_thresh": getattr(result, "lowtrust_thresh", 0.6),
        "trust_hist_pct": [round(x, 2) for x in getattr(result, "trust_hist", [])],
        "fusion_weights": dict(zip(("w_residual", "w_uncertainty"),
                                   getattr(result, "w_used", (0.6, 0.4)))),
        "n_ensemble": result.n_ensemble,
        "step_skip": result.step_skip,
        "elapsed_s": round(result.elapsed_s, 3),
        "recursions_per_step_first_member": (result.recursions_used[0]
                                             if result.recursions_used else []),
        "step_residual_curve_first_member": (result.step_residuals[0]
                                             if result.step_residuals else []),
        "quality_metrics_official_protocol": metrics or None,
        "disclaimer": "合成结果仅供科研参考，不能直接用于临床诊断决策；"
                      "低信任区域提示模型可能产生幻觉，建议人工复核。",
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
