"""把离线演示包与本地源码严格对齐，并做逐文件一致性校验。

原则：**别人能跑到的 = 本地能跑到的**。
- 代码：app/ trustbridge/ third_party/SelfRDB 全量覆盖
- 权重：自动取 engine.TASKS 里的全部 ckpt（新增任务会自动带上，不会漏）
- 资源：门控/强度模板 npz、示例切片
- 文档：README + 应用使用手册（放 docs_demo/）

用法:
  python scripts/sync_demo_package.py              # 同步 + 校验
  python scripts/sync_demo_package.py --check-only # 只校验，不改文件
  python scripts/sync_demo_package.py --no-docs    # 不带文档
"""
import argparse
import hashlib
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist", "MedBridgeStudio_Demo")
sys.path.insert(0, ROOT)

SKIP_DIRS = {"__pycache__", ".git", ".parts", "parts", "figures", "__MACOSX"}
SKIP_EXT = (".pyc", ".pyo", ".tmp", ".log")


def sha1(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def copy_tree(src, dst):
    """覆盖式复制；同时清掉目标里源已不存在的文件，保证两侧一致。"""
    if not os.path.isdir(src):
        return 0
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel = os.path.relpath(root, src)
        out = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(out, exist_ok=True)
        keep = set()
        for f in files:
            if f.endswith(SKIP_EXT):
                continue
            s, d = os.path.join(root, f), os.path.join(out, f)
            keep.add(f)
            if (not os.path.exists(d)) or os.path.getsize(s) != os.path.getsize(d):
                shutil.copy2(s, d)
                n += 1
        # 目标里多余的文件删掉
        if os.path.isdir(out):
            for f in os.listdir(out):
                if f not in keep and os.path.isfile(os.path.join(out, f)):
                    os.remove(os.path.join(out, f))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="只校验不一致，不复制")
    ap.add_argument("--no-docs", action="store_true", help="不附带文档")
    args = ap.parse_args()

    # 权重清单直接从 TASKS 取 —— 保证"本地能跑的任务 = 包里能跑的任务"
    from trustbridge.engine import TASKS
    ckpts = sorted({ck for (_d, _s, _t, ck) in TASKS.values()})

    pairs = []  # (本地路径, 包内相对路径)
    for d in ("app", "trustbridge", os.path.join("third_party", "SelfRDB")):
        src = os.path.join(ROOT, d)
        for root, dirs, files in os.walk(src):
            dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
            for f in files:
                if f.endswith(SKIP_EXT):
                    continue
                p = os.path.join(root, f)
                pairs.append((p, os.path.relpath(p, ROOT)))
    for c in ckpts:
        pairs.append((os.path.join(ROOT, "checkpoints", c), os.path.join("checkpoints", c)))
    for n in ("modality_gate.npz", "modality_gate_pelvis.npz", "intensity_template.npz"):
        pairs.append((os.path.join(ROOT, "data", n), os.path.join("data", n)))
    if not args.no_docs:
        pairs.append((os.path.join(ROOT, "README.md"), os.path.join("docs_demo", "README.md")))
        pairs.append((os.path.join(ROOT, "docs", "04_应用使用手册.md"),
                      os.path.join("docs_demo", "应用使用手册.md")))

    print("== 1) 同步 %d 个文件（代码 + %d 个权重 + 资源 + 文档）==" % (len(pairs), len(ckpts)))
    missing_src, copied = [], 0
    if not args.check_only:
        for d in ("app", "trustbridge", os.path.join("third_party", "SelfRDB")):
            copied += copy_tree(os.path.join(ROOT, d), os.path.join(DIST, d))
        os.makedirs(os.path.join(DIST, "checkpoints"), exist_ok=True)
        for c in ckpts:
            s = os.path.join(ROOT, "checkpoints", c)
            d = os.path.join(DIST, "checkpoints", c)
            if not os.path.exists(s):
                missing_src.append(c)
                continue
            if (not os.path.exists(d)) or os.path.getsize(s) != os.path.getsize(d):
                shutil.copy2(s, d)
                copied += 1
        os.makedirs(os.path.join(DIST, "data"), exist_ok=True)
        for n in ("modality_gate.npz", "modality_gate_pelvis.npz", "intensity_template.npz"):
            s, d = os.path.join(ROOT, "data", n), os.path.join(DIST, "data", n)
            if os.path.exists(s):
                shutil.copy2(s, d)
        if not args.no_docs:
            os.makedirs(os.path.join(DIST, "docs_demo"), exist_ok=True)
            shutil.copy2(os.path.join(ROOT, "README.md"),
                         os.path.join(DIST, "docs_demo", "README.md"))
            shutil.copy2(os.path.join(ROOT, "docs", "04_应用使用手册.md"),
                         os.path.join(DIST, "docs_demo", "应用使用手册.md"))
        print("   复制/更新: %d 个" % copied)
    if missing_src:
        print("   ⚠ 本地缺失（未同步）: %s" % ", ".join(missing_src))

    print("== 2) 逐文件 SHA1 一致性校验 ==")
    bad = []
    for i, (s, rel) in enumerate(pairs, 1):
        d = os.path.join(DIST, rel)
        if not os.path.exists(s):
            bad.append((rel, "本地缺失"))
            continue
        if not os.path.exists(d):
            bad.append((rel, "包内缺失"))
            continue
        if os.path.getsize(s) != os.path.getsize(d) or sha1(s) != sha1(d):
            bad.append((rel, "内容不一致"))
        if i % 40 == 0:
            print("   ...%d/%d" % (i, len(pairs)), flush=True)
    print("   校验 %d 个文件，不一致 %d 个" % (len(pairs), len(bad)))
    for rel, why in bad:
        print("   [X] %-46s %s" % (rel, why))

    print("== 3) 任务可用性（包内权重 vs TASKS 声明）==")
    ok = [t for t, (_d, _s, _x, ck) in TASKS.items()
          if os.path.exists(os.path.join(DIST, "checkpoints", ck))]
    miss = [t for t in TASKS if t not in ok]
    print("   可用任务 %d/%d: %s" % (len(ok), len(TASKS), ", ".join(ok)))
    if miss:
        print("   缺失: %s" % ", ".join(miss))

    ok_all = not bad and not miss
    print()
    print("SYNC OK: 包与本地完全一致" if ok_all else "SYNC FAILED: 见上面 [X] 与缺失项")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
