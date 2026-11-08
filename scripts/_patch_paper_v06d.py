# -*- coding: utf-8 -*-
"""v0.6.2: zh paragraph 句号→冒号；en author 块精简。"""
import io
import re

# zh paragraph 标题句号→冒号
fp = r"D:\MedTranslateProject\docs\latex\main_zh.tex"
s = io.open(fp, encoding="utf-8").read()
n = len(re.findall(r"\\paragraph\{[^{}]*?\。}", s))
s = re.sub(r"(\\paragraph\{[^{}]*?)。(\})", r"\1：\2", s)
io.open(fp, "w", encoding="utf-8").write(s)
print("zh paragraph titles -> colon:", n)

# en author 块精简
fp2 = r"D:\MedTranslateProject\docs\latex\main.tex"
s2 = io.open(fp2, encoding="utf-8").read()
old2 = "\\author{%\n  Author list to be finalized\\\\\n  Institution\\\\\n  \\small Code and weights:"
assert old2 in s2, "en author"
s2 = s2.replace(old2, "\\author{%\n  \\small Code and weights:", 1)
io.open(fp2, "w", encoding="utf-8").write(s2)
print("en author trimmed")
