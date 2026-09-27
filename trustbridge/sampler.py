"""TrustSampler —— 本项目核心创新层。

在官方 SelfRDB 反向采样过程（diffusion.DiffusionBridge.sample_x0）基础上增加：
  1. 递归收敛残差场 RCRF：捕获反向最后若干步内、相邻递归估计之间的逐像素残差，
     定位生成器"无法自洽收敛"的空间区域（幻觉候选区）；
  2. 随机桥集成不确定性：N 次独立桥采样（不同噪声种子与生成器 latent z），
     逐像素标准差作为不确定性图；
  3. 自适应递归停机：沿用官方阈值判据但记录每步实际递归次数与残差衰减曲线；
  4. 步长跳跃 (step_skip)：将相邻后验步合并以加速推理（质量-速度可调）。

数据域约定：模型域 [-1,1]，形状 [B,1,H,W]。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class TrustResult:
    pred: torch.Tensor                 # [1,1,H,W] 集成均值估计（模型域 [-1,1]）
    members: torch.Tensor              # [N,1,H,W] 各集成成员
    rcrf_map: np.ndarray               # [H,W] 递归收敛残差场（已归一化 0-1）
    unc_map: np.ndarray                # [H,W] 集成不确定性图（已归一化 0-1）
    trust_map: np.ndarray              # [H,W] 融合信任图（1=可信）
    trust_score: float                 # 0-100 图像级信任分
    qc_flag: str                       # PASS / REVIEW / REJECT
    step_residuals: list = field(default_factory=list)   # 每反向步的末次递归残差(标量)
    recursions_used: list = field(default_factory=list)  # 每反向步实际递归次数
    n_ensemble: int = 1
    step_skip: int = 1
    elapsed_s: float = 0.0


class TrustSampler:
    def __init__(self,
                 engine,
                 capture_last_steps: int = 3,
                 w_residual: float = 0.6,
                 w_uncertainty: float = 0.4,
                 qc_pass: float = 75.0,
                 qc_review: float = 55.0,
                 pct: float = 99.0):
        """
        Args:
            engine: BridgeEngine
            capture_last_steps: 捕获递归残差的最后若干个反向步
            w_residual / w_uncertainty: 信任图融合权重
            qc_pass / qc_review: 质控阈值（Trust Score）
            pct: 归一化分位（抗离群值）
        """
        self.eng = engine
        self.capture_last_steps = capture_last_steps
        self.w_residual = w_residual
        self.w_uncertainty = w_uncertainty
        self.qc_pass = qc_pass
        self.qc_review = qc_review
        self.pct = pct

    # ---------- 官方后验公式的步长泛化版 ----------
    def _q_posterior(self, t: torch.Tensor, t_prev: torch.Tensor,
                     x_t: torch.Tensor, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """采样 q(x_{t_prev} | x_t, x0, y)（t_prev 可为 0；公式与官方一致，索引泛化）。"""
        shape = [-1] + [1] * (x0.ndim - 1)
        d = self.eng.diffusion
        var_t = d.std[t].view(shape) ** 2
        var_tp = d.std[t_prev].view(shape) ** 2
        mu_x0_t = d.mu_x0[t].view(shape)
        mu_x0_tp = d.mu_x0[t_prev].view(shape)
        mu_y_t = d.mu_y[t].view(shape)
        mu_y_tp = d.mu_y[t_prev].view(shape)

        var_t_tp = var_t - var_tp * (mu_x0_t / mu_x0_tp) ** 2
        v = var_t_tp * (var_tp / var_t)

        mean = mu_x0_tp * x0 + mu_y_tp * y + \
            ((var_tp - v) / var_t).sqrt() * (x_t - mu_x0_t * x0 - mu_y_t * y)
        noise = 0.0 if float(v.flatten()[0]) == 0 else v.sqrt() * torch.randn_like(x_t)
        return mean + noise

    def _q_sample_T(self, y: torch.Tensor) -> torch.Tensor:
        """终点样本 x_T = 加噪源图像 y_ε。"""
        d = self.eng.diffusion
        T = self.eng.n_steps
        return d.q_sample(torch.tensor([T], device=y.device),
                          torch.zeros_like(y), y)

    def _block_pool(self, x: torch.Tensor, block: int) -> torch.Tensor:
        """[B,1,H,W] → [B,1,nB,nB] 的块均值。"""
        B, C, H, W = x.shape
        nb = H // block
        return torch.nn.functional.avg_pool2d(x[:, :, : nb * block, : nb * block],
                                              kernel_size=block, stride=block)

    def _block_upsample(self, blk: torch.Tensor, block: int, H: int, W: int) -> torch.Tensor:
        """[nB,nB] 或 [B,nB,nB] → [B,1,H,W]（块值重复展开，多余边缘用邻近块值）。"""
        if blk.dim() == 2:
            blk = blk[None]
        if blk.dim() == 3:
            blk = blk[:, None]
        nb = blk.shape[-1]
        up = blk.repeat_interleave(block, dim=2).repeat_interleave(block, dim=3)
        out = torch.zeros(blk.shape[0], 1, H, W, device=blk.device, dtype=blk.dtype)
        h, w = min(nb * block, H), min(nb * block, W)
        out[:, :, :h, :w] = up[:, :, :h, :w]
        if h < H:
            out[:, :, h:, :w] = up[:, :, -1:, :w]
        if w < W:
            out[:, :, :H, w:] = out[:, :, :H, -1:]
        return out

    @torch.inference_mode()
    def sample(self, y: torch.Tensor,
               n_ensemble: int = 4,
               step_skip: int = 1,
               seed: int = 0,
               batch_chunk: int = 2,
               progress_cb=None) -> TrustResult:
        """
        Args:
            y: 源图像（模型域 [-1,1]，[1,1,H,W]）
            n_ensemble: 集成采样数（1=单次，仅 RCRF 无不确定性）
            step_skip: 反向步跳跃（1=官方逐步；2/3 加速）
            seed: 随机种子
            batch_chunk: 集成成员分批前向（控显存）
        """
        assert y.ndim == 4 and y.shape[0] == 1
        dev = y.device
        t0 = time.time()
        torch.manual_seed(seed)

        T = self.eng.n_steps
        skip = max(1, int(step_skip))
        # 反向时间表：T → ... → 1（确保最后一步落到 1，后验到 x0）
        ts = list(range(T, 0, -skip))
        if ts[-1] != 1:
            ts.append(1)

        n = max(1, int(n_ensemble))
        chunks = [c for c in range(0, n, batch_chunk)] or [0]

        members_out, rcrf_acc, step_curve, rec_used = [], None, [], []
        n_chunks = len(chunks)

        for ci, c0 in enumerate(chunks):
            bs = min(batch_chunk, n - c0)
            y_rep = y.repeat(bs, 1, 1, 1)
            # 各成员独立的终点噪声 x_T = y_ε
            x_t = self.eng.diffusion.q_sample(
                torch.full((bs,), self.eng.n_steps, device=dev, dtype=torch.long),
                torch.zeros(bs, 1, y.shape[-2], y.shape[-1], device=dev), y_rep)
            rcrf_sum = None
            curve, used = [], []

            for t in ts:
                # 官方自洽递归（带逐像素残差捕获）
                x0_r = torch.zeros_like(x_t)
                r = 0
                res_last = None
                for r in range(1, self.eng.max_recursions + 1):
                    x0_rp1 = self.eng.generator(
                        torch.cat((x_t, y_rep), dim=1),
                        torch.full((bs,), t, device=dev, dtype=torch.long),
                        x_r=x0_r)
                    res_last = (x0_rp1 - x0_r).abs().mean(dim=1)   # [bs,H,W]
                    change = res_last.mean(dim=0).max().item()
                    if change < self.eng.consistency_threshold:
                        break
                    x0_r = x0_rp1

                x0_pred = x0_r if r > 1 else x0_rp1
                used.append(r)
                curve.append(float(res_last.mean().item()))

                # 捕获最后若干步的残差场（越接近数据空间越有意义）
                if t <= self.capture_last_steps and res_last is not None:
                    rcrf_sum = res_last.sum(0) if rcrf_sum is None \
                        else rcrf_sum + res_last.sum(0)

                t_prev = max(t - skip, 0)
                x_t = self._q_posterior(torch.full((bs,), t, device=dev, dtype=torch.long),
                                        torch.full((bs,), t_prev, device=dev, dtype=torch.long),
                                        x_t, x0_pred, y_rep)

                if progress_cb:
                    done = (ci + (ts.index(t) + 1) / len(ts)) / n_chunks
                    progress_cb(float(min(done, 0.99)),
                                f"集成批次 {ci+1}/{n_chunks} · 反向步 t={t}")

            members_out.append(x0_pred.detach())
            rcrf_acc = rcrf_sum if rcrf_acc is None else rcrf_acc + rcrf_sum
            step_curve.append(curve)
            rec_used.append(used)

        members = torch.cat(members_out, dim=0)              # [N,1,H,W]
        pred = members.mean(dim=0, keepdim=True)             # [1,1,H,W]

        # ---- RCRF：捕获窗内残差均值 → 分位归一化 ----
        rcrf = (rcrf_acc / (self.capture_last_steps * n)).cpu().numpy()
        rcrf_n = _pct_norm(rcrf, self.pct)

        # ---- 集成不确定性：成员间逐像素标准差 → 分位归一化 ----
        if n > 1:
            unc = members.std(dim=0)[0].cpu().numpy()   # [H,W]
        else:
            unc = np.zeros_like(rcrf)
        unc_n = _pct_norm(unc, self.pct) if n > 1 else unc

        # ---- 信任图与图像级评分 ----
        trust = np.clip(1.0 - self.w_residual * rcrf_n - self.w_uncertainty * unc_n, 0.0, 1.0)
        score = float(trust.mean() * 100.0)
        flag = "PASS" if score >= self.qc_pass else \
            ("REVIEW" if score >= self.qc_review else "REJECT")

        return TrustResult(
            pred=pred, members=members,
            rcrf_map=rcrf_n, unc_map=unc_n, trust_map=trust,
            trust_score=score, qc_flag=flag,
            step_residuals=step_curve, recursions_used=rec_used,
            n_ensemble=n, step_skip=skip,
            elapsed_s=time.time() - t0,
        )


def _pct_norm(m: np.ndarray, pct: float = 99.0) -> np.ndarray:
    """按高分位归一化到 [0,1]，抑制离群值主导。"""
    hi = np.percentile(m, pct)
    if hi <= 1e-12:
        return np.zeros_like(m)
    return np.clip(m / hi, 0.0, 1.0)


class AdaptiveTrustSampler(TrustSampler):
    """信任图引导的自适应采样（改进阶段一）。

    机制 A 块级自适应递归：把图像划成 block×block 个图块，每个反向步内递归时
      已收敛块冻结其估计、未收敛块继续精化，至多 r_max 次——难块获得更多修正预算。
    机制 B 集成预算再分配：先跑 n_base 条链得初步信任图，仅对低信任块追加 m_extra 条链，
      低信任块用追加链均值替换（总预算 ≤ n_base+m_extra，与均匀基线对齐可比）。
    """

    @torch.inference_mode()
    def sample_adaptive(self, y: torch.Tensor,
                        n_base: int = 2, m_extra: int = 2,
                        r_max: int = 4, block_thr: float = 0.01,
                        block: int = 4, step_skip: int = 1,
                        seed: int = 0, extra_threshold: float | None = None,
                        progress_cb=None) -> TrustResult:
        """
        Args:
            n_base/m_extra: 基础链数与低信任块追加链数（总预算 ≤ n_base+m_extra）
            r_max: 块级自适应递归的最大次数
            block_thr: 块级递归收敛阈值（块内平均残差）
            block: 图块边长（block×block 像素）
            extra_threshold: 判定"低信任块"的初步信任阈值；None=取最低 30% 的块
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
        H, W = y.shape[-2:]

        chains = []
        used_extra = False
        total_chains = n_base + m_extra

        # ---------- 阶段 1：n_base 条块级自适应链 ----------
        for c in range(n_base):
            x_t = self.eng.diffusion.q_sample(
                torch.full((1,), T, device=dev, dtype=torch.long),
                torch.zeros(1, 1, H, W, device=dev), y)
            y_rep = y
            rcrf_sum, curve, used = None, [], []
            for t in ts:
                tt = torch.full((1,), t, device=dev, dtype=torch.long)
                x0_r = torch.zeros_like(x_t)
                best_res = None                     # 每块历史最优的自洽残差
                res_last = None
                r = 0
                for r in range(1, r_max + 1):
                    x0_rp1 = self.eng.generator(
                        torch.cat((x_t, y_rep), dim=1),
                        torch.full((1,), t, device=dev, dtype=torch.long),
                        x_r=x0_r)
                    res_last = (x0_rp1 - x0_r).abs().mean(dim=1)      # [1,H,W]
                    block_res = self._block_pool(res_last[:, None], block)[:, 0]
                    if best_res is None:
                        take = torch.ones_like(block_res, dtype=torch.bool)
                    else:
                        take = block_res < best_res                  # 块级择优
                    best_res = (block_res if best_res is None else
                                torch.minimum(best_res, block_res))
                    take_px = self._block_upsample(
                        take.float(), block, H, W).bool()
                    x0_r = torch.where(take_px, x0_rp1, x0_r)
                    if bool((best_res < block_thr).all()):
                        break
                used.append(r)
                curve.append(float(res_last.mean()))
                if t <= self.capture_last_steps:
                    rcrf_sum = res_last if rcrf_sum is None else rcrf_sum + res_last
                t_prev = max(t - skip, 0)
                x_t = self._q_posterior(tt, torch.full((1,), t_prev, device=dev,
                                                       dtype=torch.long),
                                        x_t, x0_r, y_rep)
                if progress_cb:
                    progress_cb((c + (ts.index(t) + 1) / len(ts)) / total_chains,
                                f"基础链 {c+1}/{n_base} · t={t}")
            chains.append({"x0": x0_r, "rcrf": rcrf_sum, "curve": curve, "used": used})

        # ---------- 阶段 2：块级初步信任 ----------
        x0s = torch.stack([c["x0"][0] for c in chains], dim=0)      # [n,1,H,W]
        rcrfs = torch.stack([c["rcrf"] for c in chains], dim=0)      # [n,1,H,W]
        block_std = self._block_pool(x0s.std(dim=0, keepdim=True), block)          # 链间分歧（块级）
        block_res = self._block_pool(rcrfs.mean(dim=0, keepdim=True), block)       # 残差（块级）
        bs_n = block_std / (block_std.max() + 1e-8)
        br_n = block_res / (block_res.max() + 1e-8)
        prelim = 1 - (self.w_residual * br_n + self.w_uncertainty * bs_n)   # [1,1,nB,nB]

        nb = prelim.shape[-1]
        if extra_threshold is None:
            thr_v = torch.quantile(prelim.flatten(), 0.3)            # 最低 30% 块
        else:
            thr_v = torch.tensor(float(extra_threshold), device=dev)
        low_blocks = prelim < thr_v

        # ---------- 阶段 3：低信任块追加链 ----------
        x0_extra_list = []
        run_extra = bool(low_blocks.any()) and m_extra > 0
        if run_extra:
            used_extra = True
            for c in range(m_extra):
                x_t = self.eng.diffusion.q_sample(
                    torch.full((1,), T, device=dev, dtype=torch.long),
                    torch.zeros(1, 1, H, W, device=dev), y)
                y_rep = y
                for t in ts:
                    tt = torch.full((1,), t, device=dev, dtype=torch.long)
                    x0_r = torch.zeros_like(x_t)
                    for _ in range(self.eng.max_recursions):
                        x0_r = self.eng.generator(torch.cat((x_t, y_rep), dim=1), tt,
                                                  x_r=x0_r)
                    x_t = self._q_posterior(tt, torch.full((1,), max(t - skip, 0),
                                        device=dev, dtype=torch.long), x_t, x0_r, y_rep)
                    if progress_cb:
                        progress_cb((n_base + (c + 1) / m_extra) / total_chains,
                                    f"追加链 {c+1}/{m_extra} · t={t}")
                x0_extra_list.append(x0_r[0])
        if run_extra:
            x0_extra = torch.stack(x0_extra_list, dim=0)             # [m,1,H,W]
            low_px = self._block_upsample(low_blocks.float(), block, H, W).bool()
            extra_mean = x0_extra.mean(dim=0, keepdim=True)
            base_mean = x0s.mean(dim=0, keepdim=True)
            pred = torch.where(low_px, extra_mean, base_mean)        # 仅低信任块替换
        else:
            x0_extra = None
            pred = x0s.mean(dim=0, keepdim=True)
            low_px = torch.zeros_like(pred, dtype=torch.bool)

        members = torch.cat([x0s] + ([x0_extra] if x0_extra is not None else []), dim=0)

        # ---------- 指标图（像素级） ----------
        rcrf_px = (rcrfs.mean(dim=0)).cpu().numpy()
        if x0_extra is not None:
            unc_px = members.std(dim=0)[0].cpu().numpy()
        else:
            unc_px = x0s.std(dim=0)[0].cpu().numpy()
        rcrf_n = _pct_norm(rcrf_px, self.pct)
        unc_n = _pct_norm(unc_px, self.pct) if x0_extra is not None else np.zeros_like(rcrf_n)
        trust = np.clip(1 - self.w_residual * rcrf_n - self.w_uncertainty * unc_n,
                        0.0, 1.0)
        score = float(trust.mean() * 100.0)
        flag = ("PASS" if score >= self.qc_pass else
                ("REVIEW" if score >= self.qc_review else "REJECT"))

        return TrustResult(
            pred=pred, members=members,
            rcrf_map=rcrf_n, unc_map=unc_n, trust_map=trust,
            trust_score=score, qc_flag=flag,
            step_residuals=[c["curve"] for c in chains],
            recursions_used=[c["used"] for c in chains],
            n_ensemble=members.shape[0], step_skip=skip,
            elapsed_s=time.time() - t0,
        )
