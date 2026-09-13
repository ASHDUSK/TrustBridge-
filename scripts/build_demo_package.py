"""构建零安装离线演示包：MedBridgeStudio_Demo/

结构:
  MedBridgeStudio_Demo/
    启动.bat  README_DEMO.txt
    app/  trustbridge/  third_party/SelfRDB/  checkpoints/  outputs/
    runtime/            内嵌便携 Python 3.13 + 全量 site-packages（离线）

用法:
  python scripts/build_demo_package.py            # 精简包（官方 T1<->T2 + 自训练）
  python scripts/build_demo_package.py --all      # 全部 5 个权重（含 PD 任务）
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist", "MedBridgeStudio_Demo")
SYS_SITE = os.path.join(sys.prefix, "Lib", "site-packages")
PY_VER = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
EMBED_URLS = [
    f"https://mirrors.huaweicloud.com/python/{PY_VER}/python-{PY_VER}-embed-amd64.zip",
    f"https://www.python.org/ftp/python/{PY_VER}/python-{PY_VER}-embed-amd64.zip",
]

DEFAULT_CKPTS = ["ixi_t2_t1.ckpt", "ixi_t1_t2.ckpt", "self_t2t1.ckpt"]
ALL_CKPTS = DEFAULT_CKPTS + ["ixi_pd_t1.ckpt", "ixi_t1_pd.ckpt"]


def copy_tree(src, dst, ignore_dirs=(".git", "__pycache__", ".parts", "parts")):
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*ignore_dirs),
                    dirs_exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="包含全部官方任务权重（PD）")
    ap.add_argument("--no-runtime", action="store_true", help="跳过内嵌 Python（仅更新代码/权重）")
    args = ap.parse_args()
    ckpts = ALL_CKPTS if args.all else DEFAULT_CKPTS

    os.makedirs(DIST, exist_ok=True)
    print("== 1) 代码与资源 ==")
    copy_tree(os.path.join(ROOT, "app"), os.path.join(DIST, "app"))
    copy_tree(os.path.join(ROOT, "trustbridge"), os.path.join(DIST, "trustbridge"))
    copy_tree(os.path.join(ROOT, "third_party", "SelfRDB"),
              os.path.join(DIST, "third_party", "SelfRDB"),
              ignore_dirs=(".git", "__pycache__", "figures"))
    os.makedirs(os.path.join(DIST, "outputs"), exist_ok=True)

    print("== 2) 权重 ==")
    os.makedirs(os.path.join(DIST, "checkpoints"), exist_ok=True)
    for c in ckpts:
        src = os.path.join(ROOT, "checkpoints", c)
        dst = os.path.join(DIST, "checkpoints", c)
        if not os.path.exists(src):
            print(f"  缺少 {c}，跳过"); continue
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        print(f"  {c} ({os.path.getsize(dst)/1e6:.0f} MB)")

    if not args.no_runtime:
        print("== 3) 内嵌 Python ==")
        rt = os.path.join(DIST, "runtime")
        exe = os.path.join(rt, "python.exe")
        if not os.path.exists(exe):
            os.makedirs(rt, exist_ok=True)
            zip_path = os.path.join(ROOT, "dist", "python-embed.zip")
            if not os.path.exists(zip_path):
                ok = False
                for url in EMBED_URLS:
                    try:
                        print(f"  下载 {url}")
                        urllib.request.urlretrieve(url, zip_path)
                        ok = True
                        break
                    except Exception as e:
                        print(f"  失败: {e}")
                if not ok:
                    sys.exit("内嵌 Python 下载失败，请手动下载 python-3.13.x-embed-amd64.zip "
                             "放到 dist/python-embed.zip 后重跑")
            import zipfile
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(rt)
            # 启用 site-packages 搜索
            pth = os.path.join(rt, f"python{sys.version_info.major}{sys.version_info.minor}._pth")
            with open(pth, "a", encoding="ascii") as f:
                f.write("Lib\\site-packages\n")
        # 复制 site-packages（与打包机同版本 Python 3.13，二进制兼容）
        sp = os.path.join(rt, "Lib", "site-packages")
        if not os.path.isdir(sp):
            print(f"  复制 site-packages ({sum(os.path.getsize(os.path.join(dp,f)) for dp,_,fs in os.walk(SYS_SITE) for f in fs)/1e9:.1f} GB)...")
            copy_tree(SYS_SITE, sp)

    print("== 4) 启动脚本与说明 ==")
    bat = """@echo off
title MedBridge Studio
cd /d %~dp0
set GRADIO_ANALYTICS_ENABLED=False
set MPLCONFIGDIR=%TEMP%\\mplcache
if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"
echo Starting MedBridge Studio (first load takes 10-20 seconds)...
echo Browser will open at http://127.0.0.1:7860  (close this window to stop)
start "" cmd /c "timeout /t 12 >nul & start http://127.0.0.1:7860"
runtime\\python.exe app\\app.py
pause
"""
    with open(os.path.join(DIST, "启动.bat"), "w", encoding="ascii", newline="\r\n") as f:
        f.write(bat)
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(DIST) for f in fs)
    print(f"\n完成: {DIST}\n总大小 {total/1e9:.1f} GB")


if __name__ == "__main__":
    main()
