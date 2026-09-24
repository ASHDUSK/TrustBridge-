"""TrustBridge 引擎层：加载 SelfRDB 官方预训练权重，剥离 Lightning 依赖。

官方仓库: https://github.com/icon-lab/SelfRDB (MIT License)
论文: Arslan et al., "Self-Consistent Recursive Diffusion Bridge for
      Medical Image Translation", arXiv:2405.06789
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# 挂载官方代码路径（diffusion.py / backbones/）
REPO_ROOT = Path(__file__).resolve().parent.parent
SELF_RDB_DIR = REPO_ROOT / "third_party" / "SelfRDB"
if str(SELF_RDB_DIR) not in sys.path:
    sys.path.insert(0, str(SELF_RDB_DIR))

from diffusion import DiffusionBridge          # noqa: E402  (官方模块)
from backbones.ncsnpp import NCSNpp            # noqa: E402

DEFAULT_CKPT_DIR = REPO_ROOT / "checkpoints"

# 支持的任务: (任务ID, 显示名, 源模态, 目标模态, ckpt 文件名)
TASKS = {
    "T1->T2":  ("T1 → T2（脑MRI）", "T1", "T2", "ixi_t1_t2.ckpt"),
    "T2->T1":  ("T2 → T1（脑MRI）", "T2", "T1", "ixi_t2_t1.ckpt"),
    "PD->T1":  ("PD → T1（脑MRI）", "PD", "T1", "ixi_pd_t1.ckpt"),
    "T1->PD":  ("T1 → PD（脑MRI）", "T1", "PD", "ixi_t1_pd.ckpt"),
    # 自训练模型（128x128，本仓库全流程可复现）
    "T2->T1*": ("T2 → T1（自训练·128px）", "T2", "T1", "self_t2t1.ckpt"),
    "T1->T2*": ("T1 → T2（自训练·128px）", "T1", "T2", "self_t1t2.ckpt"),
    # BRATS 脑肿瘤（官方预训练 256²）
    "BRATS_T2->T1":    ("BRATS T2 → T1（脑肿瘤MRI）", "T2", "T1", "brats_t2_t1.ckpt"),
    "BRATS_FLAIR->T2": ("BRATS FLAIR → T2（脑肿瘤MRI）", "FLAIR", "T2", "brats_flair_t2.ckpt"),
    # 盆腔 MRI→CT（官方预训练 256²）
    "CT_T1->CT": ("盆腔 T1 → CT（sCT）", "T1", "CT", "ct_t1_ct.ckpt"),
    "CT_T2->CT": ("盆腔 T2 → CT（sCT）", "T2", "CT", "ct_t2_ct.ckpt"),
}


class BridgeEngine:
    """单任务推理引擎：只保留生成器 Gθ 与扩散调度，不含判别器/训练逻辑。"""

    def __init__(self, task_id: str, ckpt_path: str | None = None,
                 device: str | None = None):
        task_id = task_id.replace("→", "->").replace(" ", "")
        if task_id not in TASKS:
            raise ValueError(f"未知任务: {task_id}，可选: {list(TASKS)}")
        self.task_id = task_id
        self.display, self.source_mod, self.target_mod, ckpt_name = TASKS[task_id]

        if ckpt_path is None:
            ckpt_path = str(DEFAULT_CKPT_DIR / ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"权重不存在: {ckpt_path}\n"
                f"请先运行 scripts/download_checkpoints.sh 下载官方预训练权重")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        hparams = blob.get("hyper_parameters", {})
        gen_params = dict(hparams.get("generator_params", {}))
        diff_params = dict(hparams.get("diffusion_params", {}))
        if not gen_params or not diff_params:
            raise RuntimeError("ckpt 中缺少 hyper_parameters，无法构建网络")

        # 生成器 Gθ
        self.generator = NCSNpp(**gen_params).to(self.device).eval()
        g_state = {k[len("generator."):]: v for k, v in blob["state_dict"].items()
                   if k.startswith("generator.")}
        self.generator.load_state_dict(g_state, strict=True)

        # 扩散调度：ckpt 内保存的 buffer 是训练时真实调度，直接采用；
        # 未保存的 buffer 退回超参重建值（个别权重如 ct_t2_ct 的重建值与保存值不符，
        # 一律以 ckpt 保存值为准）
        self.diffusion = DiffusionBridge(**diff_params).to(self.device)
        d_state = {k[len("diffusion."):]: v for k, v in blob["state_dict"].items()
                   if k.startswith("diffusion.")}
        if d_state:
            self.diffusion.load_state_dict(d_state, strict=False)

        self.n_steps = int(diff_params.get("n_steps", 10))
        self.max_recursions = int(diff_params.get("n_recursions", 2))
        self.consistency_threshold = float(diff_params.get("consistency_threshold", 0.01))
        self.image_size = int(gen_params.get("image_size", 256))

        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True

    @property
    def param_count_m(self) -> float:
        return sum(p.numel() for p in self.generator.parameters()) / 1e6

    # ---------- 预处理 / 后处理（与官方 datasets.py 约定一致） ----------
    def preprocess(self, img01: np.ndarray) -> torch.Tensor:
        """[0,1] float32 HxW -> [-1,1] 1x1xSxS tensor（过大则缩放，过小则居中补零）。"""
        img = img01.astype(np.float32)
        H, W = img.shape
        S = self.image_size
        if (H, W) != (S, S):
            if H > S or W > S:
                from skimage.transform import resize
                img = resize(img, (S, S), order=1, preserve_range=True).astype(np.float32)
            else:
                canvas = np.zeros((S, S), dtype=np.float32)
                pt, pl = (S - H) // 2, (S - W) // 2
                canvas[pt:pt + H, pl:pl + W] = img
                img = canvas
        x = torch.from_numpy(img * 2.0 - 1.0)[None, None].to(self.device)
        return x

    def postprocess(self, t: torch.Tensor, orig_hw: tuple[int, int] | None = None) -> np.ndarray:
        """[-1,1] tensor -> [0,1] HxW numpy（超过模型尺寸时缩放回原尺寸，否则裁补零）。"""
        img = t.detach().float().clamp(-1, 1).squeeze().cpu().numpy() * 0.5 + 0.5
        if orig_hw is not None and img.shape != tuple(orig_hw):
            H, W = orig_hw
            S = img.shape[0]
            if H > S or W > S:
                from skimage.transform import resize
                img = resize(img, (H, W), order=1, preserve_range=True).astype(np.float32)
            else:
                pt, pl = (S - H) // 2, (S - W) // 2
                img = img[pt:pt + H, pl:pl + W]
        return np.clip(img, 0.0, 1.0)
