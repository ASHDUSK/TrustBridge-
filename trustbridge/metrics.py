"""图像质量指标：与官方 SelfRDB utils.compute_metrics 完全同口径。

官方口径：mean_norm(|x|/mean|x|) 后，skimage PSNR(data_range=gt.max())，
SSIM(data_range=gt.max())×100。
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def mean_norm(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    return x / x.mean(axis=(-1, -2), keepdims=True)


def psnr_ssim(gt01: np.ndarray, pred01: np.ndarray) -> tuple[float, float]:
    """输入 [0,1] HxW 图像对，返回 (PSNR dB, SSIM %)（官方口径）。"""
    gt = mean_norm(gt01.astype(np.float64))
    pr = mean_norm(pred01.astype(np.float64))
    p = peak_signal_noise_ratio(gt, pr, data_range=gt.max())
    s = structural_similarity(gt, pr, data_range=gt.max()) * 100
    return float(p), float(s)


def error_map(gt01: np.ndarray, pred01: np.ndarray) -> np.ndarray:
    """绝对误差图 [0,1]（用于与信任图对照验证）。"""
    return np.abs(gt01.astype(np.float32) - pred01.astype(np.float32))
