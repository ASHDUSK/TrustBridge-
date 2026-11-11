# -*- coding: utf-8 -*-
"""v0.8.3 数字同步补丁：从 2026-11-12 全量重算产物读取数值，回填中英双稿。

用法:
  python scripts/_patch_v083_sync.py          # dry：只打印计划替换与锚点核对
  python scripts/_patch_v083_sync.py --apply  # 应用
"""
import argparse
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _patch_v083_sync_repl import collect, build_repl  # noqa: E402

ROOT = r"D:\MedTranslateProject"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    V = collect()
    E, Z = build_repl(V)
    print(f"\nEN replacements: {len(E)}, ZH: {len(Z)}")
    for path, repl, tag in [(os.path.join(ROOT, r"docs\latex\main.tex"), E, "EN"),
                            (os.path.join(ROOT, r"docs\latex\main_zh.tex"), Z, "ZH")]:
        s = io.open(path, encoding="utf-8").read()
        n_ok, n_missing = 0, 0
        for old, new, t in repl:
            if old == "__TAB2__":
                mm = re.search(r"\\begin\{tabular\}\{lcccccc\}.*?\\end\{tabular\}", s, re.S)
                if not mm:
                    print(f"[{tag}/{t}] TABULAR NOT FOUND")
                    n_missing += 1
                    continue
                if a.apply:
                    s = s[:mm.start()] + new + s[mm.end():]
                n_ok += 1
                continue
            cnt = s.count(old)
            if cnt != 1:
                print(f"[{tag}/{t}] count={cnt}: {old[:70]!r}")
                n_missing += 1
                continue
            if a.apply:
                s = s.replace(old, new, 1)
            n_ok += 1
        if a.apply:
            io.open(path, "w", encoding="utf-8").write(s)
        print(f"[{tag}] ok={n_ok} missing={n_missing} applied={a.apply}")


if __name__ == "__main__":
    main()
