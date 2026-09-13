"""推理管线：单切片 / 整卷翻译（供应用与验证脚本复用）。"""
from __future__ import annotations

import numpy as np

from .sampler import TrustSampler
from .metrics import psnr_ssim


def get_sampler(engine) -> TrustSampler:
    if not hasattr(engine, "_sampler"):
        engine._sampler = TrustSampler(engine)
    return engine._sampler


def translate_single(engine, img01: np.ndarray, n_ensemble: int = 4,
                     step_skip: int = 1, seed: int = 0,
                     gt01: np.ndarray | None = None,
                     progress_cb=None) -> dict:
    """单切片翻译。img01/gt01 均为 [0,1] HxW。返回统一结果字典。"""
    sampler = get_sampler(engine)
    y = engine.preprocess(img01)
    res = sampler.sample(y, n_ensemble=n_ensemble, step_skip=step_skip,
                         seed=seed, progress_cb=progress_cb)
    pred01 = engine.postprocess(res.pred, img01.shape)
    # 信任/不确定性/残差图与输出对齐（输入被缩放时模型分辨率 != 原分辨率）
    if res.trust_map.shape != img01.shape:
        from skimage.transform import resize
        def _rs(m):
            return resize(m, img01.shape, order=1, preserve_range=True).astype(np.float32)
        res.rcrf_map, res.unc_map, res.trust_map = _rs(res.rcrf_map), _rs(res.unc_map), _rs(res.trust_map)
    metrics = None
    if gt01 is not None and gt01.shape == pred01.shape:
        p, s = psnr_ssim(gt01, pred01)
        metrics = {"psnr": round(p, 2), "ssim": round(s, 2)}
    return {"pred01": pred01, "result": res, "metrics": metrics}


def translate_volume(engine, vol01: np.ndarray, n_ensemble: int = 1,
                     step_skip: int = 2, seed: int = 0,
                     min_frac: float = 0.02, max_slices: int | None = None,
                     progress_cb=None) -> dict:
    """整卷翻译。vol01 [0,1] HxWxZ。无脑切片直接沿用源切片。

    返回 {"out": [0,1] HxWxZ, "slices": [(z, score), ...], "preds": [HxW...]}
    """
    sampler = get_sampler(engine)
    H, W, Z = vol01.shape
    out = vol01.copy()
    zs = [k for k in range(Z) if (vol01[:, :, k] > 0.05).mean() > min_frac]
    if max_slices:
        zs = zs[:max_slices]
    scores, preds = [], {}
    for i, k in enumerate(zs):
        img = np.ascontiguousarray(vol01[:, :, k])
        y = engine.preprocess(img)
        res = sampler.sample(y, n_ensemble=n_ensemble, step_skip=step_skip, seed=seed)
        pred = engine.postprocess(res.pred, img.shape)
        out[:, :, k] = pred
        scores.append((k, res.trust_score))
        preds[k] = pred
        if progress_cb:
            progress_cb((i + 1) / len(zs), f"切片 {i+1}/{len(zs)} (z={k})")
    return {"out": out, "slices": scores, "preds": preds}
