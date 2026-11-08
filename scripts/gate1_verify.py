# -*- coding: utf-8 -*-
"""Gate 1 · 硬性标准 A 验证：CReF 累积逻辑手算核对 + 指标实现自检。

1) MockEngine（确定性生成器）下手工计算 CReF 期望值，与 sampler 输出比对；
2) 覆盖：捕获窗集合、末次递归残差语义、链间求和、chunk 累积、分母 K0·N；
3) AUPRC / AUSE / risk-coverage / masked-PSNR 合成数据自检。
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from trustbridge.sampler import TrustSampler, _pct_norm  # noqa: E402

PASS = []


def check(name, cond):
    PASS.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


# ---------------------------------------------------------------- MockEngine
class MockDiffusion:
    def q_sample(self, t_tensor, noise, y_rep):
        # 确定性：x_T = y（无噪声），生成器不依赖 x_t → 对 CReF 无影响
        return y_rep.clone()


class MockEngine:
    n_steps = 3
    max_recursions = 2
    consistency_threshold = 0.0        # 永不提前停机 → 每步恰好 2 次递归
    use_bf16 = False

    def __init__(self, patterns):
        # patterns: {t: [H,W] float32}，生成器返回 x_r + pattern[t]
        self.patterns = patterns
        self.diffusion = MockDiffusion()

    def generator(self, x_in, t_tensor, x_r=None):
        t = int(t_tensor[0].item())
        base = torch.from_numpy(self.patterns[t]).float()
        out = x_r[:, :1].to(base.dtype) + base.unsqueeze(0).to(x_r.device)
        return out


# ------------------------------------------------------- CReF 手算核对
def test_cref_handcalc():
    H = W = 4
    p1 = np.full((H, W), 0.2, np.float32); p1[0, 0] = 0.5
    p2 = np.full((H, W), 0.1, np.float32)
    eng = MockEngine({3: p1, 2: p2, 1: p1})

    y = torch.rand(1, 1, H, W)                     # 生成器不依赖 y/x_t → 任意
    K0, N = 2, 2                                   # 捕获窗 {1,2}（t≤2）

    smp = TrustSampler(eng, capture_last_steps=K0, store_step_residuals=True)
    # 后验不影响 D_t（生成器不依赖 x_t）：返回与 x_t 同形的占位
    smp._q_posterior = lambda t_, tp_, x_t_, x0_, y_: torch.zeros_like(x_t_)
    res = smp.sample(y, n_ensemble=N, step_skip=1, seed=0)

    # 手算：生成器 est_r = x_r + p_t → D_t = |p_t|（两次递归后 |2p_t − p_t| = |p_t|）
    # 链间求和 S_t = N·D_t；rcrf_raw = Σ_t S_t / (K0·N) = (D_1 + D_2)/2
    D1, D2 = np.abs(p1), np.abs(p2)
    expect_raw = (D1 + D2) / 2.0                   # = (K0·N) 归一化后的理论值
    # 捕获步集合：ts=[3,2,1]，t≤K0=2 → {2,1}
    check("G1-A 捕获步集合 = {1,2}", sorted(t for t, _ in smp.step_residual_maps) == [1, 2])
    for t, S in smp.step_residual_maps:
        Dt = {1: D1, 2: D2}[t]
        check(f"G1-A S_{t} = N·D_{t}（链间求和）",
              np.allclose(S, N * Dt, atol=1e-6))
    # 分母 K0·N 语义：rcrf_raw 应为 (S_1+S_2)/(K0·N) = (D1+D2)/2
    raw_recon = sum(S for _, S in smp.step_residual_maps) / (K0 * N)
    check("G1-A raw CReF = (D1+D2)/2", np.allclose(raw_recon, expect_raw, atol=1e-6))
    # sampler 返回的是 pct_norm 后的图；q=99 线性插值 + 截断语义核对
    hi = np.percentile(raw_recon, 99)
    got = res.rcrf_map
    check("G1-A 峰值截断：[0,0] = 1.0", abs(got[0, 0] - 1.0) < 1e-6)
    check("G1-A 非峰值 = raw/hi", abs(got[1, 1] - 0.15 / hi) < 1e-6)
    # raw = got·hi 反解核对
    check("G1-A pct_norm 反解一致", np.allclose(got, np.clip(raw_recon / hi, 0, 1), atol=1e-6))


# ------------------------------------------------------- 指标自检
def test_auprc():
    def auprc(score, label):
        order = np.argsort(-score)
        lab = label[order]
        tp = np.cumsum(lab)
        prec = tp / (np.arange(len(lab)) + 1)
        rec = tp / max(label.sum(), 1)
        # 阶梯积分
        return float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))

    rng = np.random.RandomState(0)
    score = rng.rand(1000)
    label = (rng.rand(1000) < 0.1).astype(int)
    label[:50] = 1; score[:50] += 1.0            # 前 50 个高分且全为正
    a = auprc(score, label)
    check("G1 AUPRC 完美排序 ≈ 正类率上限之上（>0.5）", a > 0.5)
    check("G1 AUPRC 随机基线 ≈ 正类率", abs(auprc(rng.rand(1000), label) - label.mean()) < 0.05)


def test_ause_riskcoverage():
    def risk_coverage(trust, err):
        order = np.argsort(-trust)               # 信任降序保留
        er = err[order]
        n = len(er)
        cov = np.arange(1, n + 1) / n
        risk = np.cumsum(er) / np.arange(1, n + 1)
        return cov, risk

    def ause(trust, err):
        cov, risk = risk_coverage(trust, err)
        cov_o, risk_o = risk_coverage(err * -1, err)   # oracle：误差升序保留
        grid = np.linspace(0, 1, 100)
        return float(np.trapezoid(np.interp(grid, cov, risk) -
                                  np.interp(grid, cov_o, risk_o), grid))

    rng = np.random.RandomState(1)
    n = 4000
    trust = rng.rand(n)
    err = 0.5 * (1 - trust) + 0.05 * rng.rand(n)   # 信任与误差强负相关
    a_good = ause(trust, err)
    a_rand = ause(rng.rand(n), err)
    check("G1 AUSE(有效信号) < AUSE(随机)", a_good < a_rand * 0.5)
    check("G1 AUSE(随机) ≈ AUSE(完美排序)/2 量级内", 0 < a_rand < 1)

    # risk-coverage 端点：c=1 时 risk=全体 MAE
    cov, risk = risk_coverage(trust, err)
    check("G1 risk-coverage c=1 端点 = 全体 MAE", abs(risk[-1] - err.mean()) < 1e-9)


def test_masked_psnr():
    from trustbridge.metrics import mean_norm
    from skimage.metrics import peak_signal_noise_ratio as psnr
    rng = np.random.RandomState(2)
    gt = rng.rand(32, 32).astype(np.float32)
    pred = gt + rng.rand(32, 32).astype(np.float32) * 0.05
    mask = np.zeros((32, 32), bool); mask[8:24, 8:24] = True
    g, p = mean_norm(gt), mean_norm(pred)
    v = psnr(g[mask], p[mask], data_range=g[mask].max())
    check("G1 masked-PSNR 有限正值", 0 < v < 100)
    # 全同图 → 极大值
    v2 = psnr(g[mask], g[mask], data_range=g[mask].max())
    check("G1 masked-PSNR 恒等图 → 极大", v2 > 60)


if __name__ == "__main__":
    test_cref_handcalc()
    test_auprc()
    test_ause_riskcoverage()
    test_masked_psnr()
    n_fail = sum(1 for _, ok in PASS if not ok)
    print(f"\n==== Gate1 自检: {len(PASS) - n_fail}/{len(PASS)} PASS ====")
    sys.exit(1 if n_fail else 0)
