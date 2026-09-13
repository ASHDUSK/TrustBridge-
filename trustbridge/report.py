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


def compose_report_png(source01, pred01, result, path, gt01=None,
                       low_thresh: float = 0.6,
                       source_mod: str = "源", target_mod: str = "目标"):
    """2×3 报告图：源 | 翻译 | 信任热图 | 叠加 | 误差(可选) | 收敛曲线。"""
    n_err = gt01 is not None
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2))
    fig.suptitle("TrustBridge 可信翻译质控报告", fontsize=17, fontweight="bold", y=0.985)

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
             f"Trust Score = {result.trust_score:.1f}/100   ·   质控结论: "
             f"{result.qc_flag}（{FLAG_TEXT.get(result.qc_flag, '')}）   ·   "
             f"集成 N={result.n_ensemble} · 步跳={result.step_skip} · 耗时 {result.elapsed_s:.1f}s",
             ha="center", fontsize=12.5, color=flag_c, fontweight="bold")
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
        "qc_flag": result.qc_flag,
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
