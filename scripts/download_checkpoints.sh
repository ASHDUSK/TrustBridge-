#!/bin/bash
# 下载模型权重（官方 SelfRDB 预训练 + TrustBridge 自训练）
#
# 用法:
#   bash scripts/download_checkpoints.sh            # 核心任务（官方 T1<->T2 + 自训练 T2->T1）
#   MIRROR=https://ghproxy.net bash scripts/download_checkpoints.sh   # 国内直连 GitHub 失败时启用镜像前缀
#   bash scripts/download_checkpoints.sh --all      # 全部官方任务（含 PD）
#
# 官方权重来自原作者 Release（MIT License）:
#   https://github.com/icon-lab/SelfRDB/releases/tag/v1.0.0
# 自训练权重来自本仓库 Release v0.1.0 附件（若 404 请见 Release 页面网盘备用链接）。
set -u
cd "$(dirname "$0")/.."
MIRROR="${MIRROR:-}"
GH="https://github.com"
[ -n "$MIRROR" ] && GH="${MIRROR}/https://github.com"
SELF_REPO="${SELF_REPO:-https://github.com/ASHDUSK/TrustBridge-}"   # ← 发布时替换为你的仓库地址

download() {  # url out size
  local url="$1" out="$2" size="${3:-}"
  if [ -f "$out" ] && [ -z "$size" ]; then echo "SKIP $out"; return 0; fi
  if [ -f "$out" ] && [ -n "$size" ] && [ "$(stat -c%s "$out")" -eq "$size" ]; then echo "SKIP $out"; return 0; fi
  echo "下载 $out ..."
  if [ -n "$size" ]; then
    bash scripts/seg_download.sh "$url" "$size" "$out" 12 && return 0
  fi
  curl -L --retry 5 -C - -o "$out" "$url"
}

mkdir -p checkpoints

# ---- 官方 SelfRDB 预训练权重（564,076,148 字节/个）----
OFFICIAL_SIZE=564076148
download "$GH/icon-lab/SelfRDB/releases/download/v1.0.0/ixi_t2_t1.ckpt" checkpoints/ixi_t2_t1.ckpt $OFFICIAL_SIZE
download "$GH/icon-lab/SelfRDB/releases/download/v1.0.0/ixi_t1_t2.ckpt" checkpoints/ixi_t1_t2.ckpt $OFFICIAL_SIZE
if [ "${1:-}" = "--all" ]; then
  download "$GH/icon-lab/SelfRDB/releases/download/v1.0.0/ixi_pd_t1.ckpt" checkpoints/ixi_pd_t1.ckpt $OFFICIAL_SIZE
  download "$GH/icon-lab/SelfRDB/releases/download/v1.0.0/ixi_t1_pd.ckpt" checkpoints/ixi_t1_pd.ckpt $OFFICIAL_SIZE
fi

# ---- TrustBridge 自训练权重（本仓库 Release 附件）----
# URL 占位：发布时把 SELF_REPO 换成实际仓库，或直接写完整 Release 下载地址
download "$SELF_REPO/releases/download/v0.1.0/self_t2t1.ckpt" checkpoints/self_t2t1.ckpt

echo "完成。checkpoints/ 就绪。"
