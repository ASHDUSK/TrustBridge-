# -*- coding: utf-8 -*-
"""中英初稿对齐审计：引用键、章节结构、关键数字。"""
import io
import re

en = io.open(r"D:\MedTranslateProject\docs\latex\main.tex", encoding="utf-8").read()
en += io.open(r"D:\MedTranslateProject\docs\latex\related_work.tex", encoding="utf-8").read()
zh = io.open(r"D:\MedTranslateProject\docs\17_论文中文初稿.md", encoding="utf-8").read()
bib = io.open(r"D:\MedTranslateProject\docs\latex\references.bib", encoding="utf-8").read()

en_keys = set()
for m in re.finditer(r"\\cite\{([^{}]*)\}", en):
    for k in m.group(1).split(","):
        en_keys.add(k.strip())
zh_keys = set(re.findall(r"\[([a-z][a-z0-9_]+?\d{4}[a-z0-9_]*|ixi_dataset|menze2015brats|thummerer2023synthrad)\]", zh))
bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))

print(f"bib 总条目: {len(bib_keys)}")
print(f"英文稿引用: {len(en_keys)} 条")
print(f"  bib 中未被英文引用: {sorted(bib_keys - en_keys)}")
print(f"  英文引用但 bib 缺失: {sorted(en_keys - bib_keys)}")
print(f"中文稿引用: {len(zh_keys)} 条")
print(f"  英有中无: {sorted(en_keys - zh_keys)}")
print(f"  中有英无: {sorted(zh_keys - en_keys)}")

zh_secs = re.findall(r"^#{2,3} (.+)$", zh, re.M)
print("\n中文稿章节:")
for s in zh_secs:
    print("  ", s)
en_secs = re.findall(r"\\(?:subsection|section)\{([^{}]*)\}", en)
print("英文稿章节:")
for s in en_secs:
    print("  ", s[:60])
