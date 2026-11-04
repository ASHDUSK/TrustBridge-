# -*- coding: utf-8 -*-
"""中英稿引用对齐终验。"""
import io
import re

D = r"D:\MedTranslateProject\docs\latex"


def keys_of(files):
    ks = set()
    for f in files:
        s = io.open(D + "\\" + f, encoding="utf-8").read()
        for m in re.finditer(r"\\cite\{([^{}]*)\}", s):
            for k in m.group(1).split(","):
                ks.add(k.strip())
    return ks


en = keys_of(["main.tex", "related_work.tex"])
zh = keys_of(["main_zh.tex", "related_work_zh.tex"])
bib = set(re.findall(r"@\w+\{([^,]+),",
                     io.open(D + r"\references.bib", encoding="utf-8").read()))
print(f"英文引用 {len(en)} | 中文引用 {len(zh)} | bib {len(bib)}")
print("中英差集:", sorted(en ^ zh) or "完全一致")
print("bib 未被引用:", sorted(bib - en - zh) or "无")
