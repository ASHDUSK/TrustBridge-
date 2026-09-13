"""跨平台训练入口：包装 third_party/SelfRDB 的 LightningCLI。

用法（仓库根目录执行）:
  python scripts/train.py fit  --config configs/train_ixi_t2t1.yaml
  python scripts/train.py test --config configs/train_ixi_t2t1.yaml --ckpt_path runs/t2t1/<version>/checkpoints/xxx.ckpt

说明: 官方 main.py 依赖其所在目录作为导入根，因此这里切换 cwd/sys.path 后再启动。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELF_RDB = os.path.join(ROOT, "third_party", "SelfRDB")

# 配置与输出路径按仓库根解析，再切到官方目录运行
args = sys.argv[1:]
fixed = []
for a in args:
    if a.startswith("--config"):
        if "=" in a:
            k, v = a.split("=", 1)
            fixed.append(f"{k}={os.path.join(ROOT, v)}")
        else:
            fixed.append(a)
            fixed.append(os.path.join(ROOT, args[args.index(a) + 1]) if args.index(a) + 1 < len(args) else "")
    else:
        fixed.append(a)

os.chdir(SELF_RDB)
sys.path.insert(0, SELF_RDB)

from main import cli_main  # noqa: E402

sys.argv = [sys.argv[0]] + [a for a in fixed if a != ""]
cli_main()
