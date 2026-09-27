"""强度标准化后处理（Nyúl 式分位数匹配）。

把模型输出（目标模态）的脑区强度分布匹配到训练集模板的分位数曲线，
修正扩散桥输出的全局/非线性强度偏差。仅改动强度，不改动解剖结构。

用法:
    from trustbridge.intensity import load_template, harmonize01
    tmpl = load_template("data/intensity_template.npz")
    out = harmonize01(pred01, tmpl["T1"])       # pred01: [0,1] HxW（或 HxWxZ）
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TEMPLATE = os.path.join(ROOT, "data", "intensity_template.npz")


def load_template(path: str = DEFAULT_TEMPLATE) -> dict:
    z = np.load(path)
    return {"pcts": z["pcts"], "C": float(z["C"]), "mask_thr": float(z["mask_thr"]),
            "T1": z["T1_val"], "T2": z["T2_val"]}


def harmonize01(img01: np.ndarray, tmpl_val: np.ndarray,
                pcts: np.ndarray | None = None,
                mask_thr: float | None = None) -> np.ndarray:
    """将 img01（[0,1]，2D 或 3D）的脑区强度分布匹配到模板分位数曲线。"""
    pcts = np.arange(0, 101, 1.0) if pcts is None else pcts
    mask_thr = 0.02 if mask_thr is None else mask_thr
    img = img01.astype(np.float32)
    mask = img > mask_thr
    if mask.sum() < 500:
        return img
    src_q = np.percentile(img[mask], pcts)
    # 保证映射源分位数单调，避免分段插值抖动
    src_q = np.maximum.accumulate(src_q)
    dst_q = np.maximum.accumulate(tmpl_val.astype(np.float64))
    mapped = np.interp(img[mask], src_q, dst_q)
    out = img.copy()
    out[mask] = np.clip(mapped, 0.0, 1.0).astype(np.float32)
    return out


def harmonize_affine01(img01: np.ndarray, tmpl_val: np.ndarray,
                       mask_thr: float | None = None) -> np.ndarray:
    """矩匹配的仿射校正：脑区内均值/标准差对齐模板（2 参数，不放大背景噪声）。

    依据消融/误差分解：合成结果与真值的差距主要是全局线性项（a·x+b），
    用模板矩估计 a、b（推理时无真值，用训练集模板统计量替代）。
    """
    mask_thr = 0.02 if mask_thr is None else mask_thr
    img = img01.astype(np.float32)
    mask = img > mask_thr
    if mask.sum() < 500:
        return img
    mu_p, sd_p = float(img[mask].mean()), float(img[mask].std() + 1e-8)
    mu_t, sd_t = float(np.mean(tmpl_val)), float(np.std(tmpl_val) + 1e-8)
    a = sd_t / sd_p
    b = mu_t - a * mu_p
    out = img.copy()
    out[mask] = np.clip(a * img[mask] + b, 0.0, 1.0).astype(np.float32)
    return out
