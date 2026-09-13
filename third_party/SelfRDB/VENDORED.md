# Vendored: SelfRDB

本目录是 [icon-lab/SelfRDB](https://github.com/icon-lab/SelfRDB) 官方代码的 vendored 副本
（MIT License，原始许可见本目录 LICENSE），作者：F. Arslan, B. Kabas, O. Dalmaz,
M. Ozbey, T. Çukur（Bilkent University ICON Lab）。

论文：*Self-Consistent Recursive Diffusion Bridge for Medical Image Translation*,
arXiv:2405.06789。

## 与上游的差异（仅兼容性补丁）

为支持较新版本的 PyTorch / 免编译环境运行，打了两处最小补丁，
均以 `TrustBridge patch` 注释标记，可在文件内检索：

1. `backbones/op/fused_act.py`、`backbones/op/upfirdn2d.py`：
   StyleGAN2 CUDA 扩展编译失败时回退到等价的纯 PyTorch 原生实现；
2. `main.py`：启用 cudnn.benchmark 与 TF32；验证/测试路径 bf16→fp32 转换
   （新版 torch 的 `.numpy()` 不支持 BFloat16）。

请优先引用与使用上游官方仓库；本副本仅为让本项目"克隆即用"。
