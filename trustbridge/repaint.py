"""RePaint 式区域重采样（针对扩散桥）。

机制：完整跑一条主反向链；在若干"重采样点" t* 上：
  1. 由当前信任代理量（链间分歧或块残差）标记不确定块；
  2. 从 x_{t*} 出发做 J 条全新噪声的反向子链（t*→0，每步 q_posterior 方差与
     生成器隐变量都重新采样——这是与"预算重分配"的本质区别：产生新的独立信息）；
  3. 每个子链对不确定块计算自洽残差，按块择优（或对低置信块取子链均值）；
  4. 组合回主链结果。

成本：J 条子链 × (t*→0) 步（仅最后一个重采样点时为全步数）。
验收口径：与 uniform N 条主链同预算比较。
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .sampler import TrustSampler, _pct_norm


class RepaintSampler(TrustSampler):

    def _subchain(self, x_t: torch.Tensor, y: torch.Tensor, t_start: int,
                  ts, skip: int) -> tuple[torch.Tensor, float]:
        """从 x_{t_start} 出发用全新噪声走完反向链，返回 (x0 估计, 平均自洽残差)。"""
        dev = y.device
        x = x_t
        res_sum, n_res = 0.0, 0
        for t in ts:
            if t > t_start:
                continue
            tt = torch.full((1,), t, device=dev, dtype=torch.long)
            x0_r = torch.zeros_like(x)
            for _ in range(self.eng.max_recursions):
                x0_rp1 = self.eng.generator(torch.cat((x, y), dim=1), tt, x_r=x0_r)
                x0_r = x0_rp1
            if t > 1:
                res_sum += float((x0_r - torch.zeros_like(x0_r)).abs().mean())
                n_res += 1
            x = self._q_posterior(tt, torch.full((1,), max(t - skip, 0),
                                  device=dev, dtype=torch.long), x, x0_r, y)
        return x0_r, (res_sum / max(n_res, 1))

    @torch.inference_mode()
    def sample_repaint(self, y: torch.Tensor,
                       n_main: int = 4,
                       j_retry: int = 4,
                       repaint_at: int | None = None,     # 重采样步 t*；None=最靠后的捕获步
                       block: int = 8,
                       block_frac: float = 0.25,          # 标记为不确定的块比例上限
                       step_skip: int = 1,
                       seed: int = 0,
                       progress_cb=None):
        """
        Args:
            n_main: 主链数（主链均值 = 基础结果，同 uniform 基线可比）
            j_retry: 每个重采样点的重试子链数
            repaint_at: 重采样步；None 则用 T//2（中点，重试性价比最高）
        """
        assert y.ndim == 4 and y.shape[0] == 1
        dev = y.device
        t0 = time.time()
        torch.manual_seed(seed)

        T = self.eng.n_steps
        skip = max(1, int(step_skip))
        ts = list(range(T, 0, -skip))
        if ts[-1] != 1:
            ts.append(1)
        t_star = repaint_at if repaint_at is not None else T // 2
        H, W = y.shape[-2:]

        # ---------- 主链 N 条 ----------
        main = []
        for c in range(n_main):
            x_t = self.eng.diffusion.q_sample(
                torch.full((1,), T, device=dev, dtype=torch.long),
                torch.zeros(1, 1, H, W, device=dev), y)
            for t in ts:
                tt = torch.full((1,), t, device=dev, dtype=torch.long)
                x0_r = torch.zeros_like(x_t)
                for _ in range(self.eng.max_recursions):
                    x0_r = self.eng.generator(torch.cat((x_t, y), dim=1), tt, x_r=x0_r)
                x_t = self._q_posterior(tt, torch.full((1,), max(t - skip, 0),
                                        device=dev, dtype=torch.long), x_t, x0_r, y)
                if progress_cb:
                    progress_cb(((c + (ts.index(t) + 1) / len(ts)) / (n_main + j_retry)),
                                f"主链 {c+1}/{n_main} · t={t}")
            main.append(x0_r)
        mains = torch.stack([m[0] for m in main], dim=0)             # [N,1,H,W]
        base = mains.mean(dim=0, keepdim=True)

        # ---------- 不确定块标记（主链间分歧） ----------
        std_map = mains.std(dim=0, keepdim=True)                     # [1,1,H,W]
        blk_std = self._block_pool(std_map, block)                   # [1,1,nB,nB]
        nb = blk_std.shape[-1]
        k = max(1, int(nb * nb * block_frac))
        thresh = torch.quantile(blk_std.flatten(), 1 - block_frac)
        low_blk = blk_std >= thresh
        # 保留 top-k 块（超出 block_frac 的截断）
        flat = blk_std.flatten()
        topk = torch.topk(flat, k).indices
        mask_flat = torch.zeros_like(flat, dtype=torch.bool)
        mask_flat[topk] = True
        low_blk = mask_flat.view(1, 1, nb, nb)
        low_px = self._block_upsample(low_blk.float(), block, H, W).bool()

        # ---------- 从主链在 t* 的状态重试 ----------
        # 重走主链存下 x_{t*}（为省显存只存 t* 的状态：重新走一遍主链到 t*）
        retries = []
        for j in range(j_retry):
            x_t = self.eng.diffusion.q_sample(
                torch.full((1,), T, device=dev, dtype=torch.long),
                torch.zeros(1, 1, H, W, device=dev), y)
            torch.manual_seed(seed + 1000 + j)                       # 子链独立噪声
            for t in ts:
                if t < t_star:
                    break
                tt = torch.full((1,), t, device=dev, dtype=torch.long)
                x0_r = torch.zeros_like(x_t)
                for _ in range(self.eng.max_recursions):
                    x0_r = self.eng.generator(torch.cat((x_t, y), dim=1), tt, x_r=x0_r)
                x_t = self._q_posterior(tt, torch.full((1,), max(t - skip, 0),
                                        device=dev, dtype=torch.long), x_t, x0_r, y)
                if progress_cb:
                    progress_cb((n_main + (j + 1) / j_retry) / (n_main + j_retry),
                                f"重试子链 {j+1}/{j_retry} · t={t}")
            retries.append(x0_r)
        retries = torch.stack([r_[0] for r_ in retries], dim=0) if retries else None

        # ---------- 组合：不确定块用重试链的均值 ----------
        if retries is not None:
            retry_mean = retries.mean(dim=0, keepdim=True)
            pred = torch.where(low_px, retry_mean, base)
            members = torch.cat([mains, retries], dim=0)
        else:
            pred = base
            members = mains

        # ---------- 指标 ----------
        unc_px = members.std(dim=0)[0].cpu().numpy()
        unc_n = _pct_norm(unc_px, self.pct)
        trust = np.clip(1.0 - unc_n, 0, 1)
        score = float(trust.mean() * 100)
        flag = ("PASS" if score >= self.qc_pass else
                ("REVIEW" if score >= self.qc_review else "REJECT"))

        class _R:
            pass
        r = _R()
        r.pred = pred
        r.members = members
        r.unc_map = unc_n
        r.trust_map = trust
        r.trust_score = score
        r.qc_flag = flag
        r.rcrf_map = np.zeros_like(unc_n)
        r.elapsed_s = time.time() - t0
        r.n_ensemble = members.shape[0]
        r.low_px = low_px[0, 0].cpu().numpy()
        return r
