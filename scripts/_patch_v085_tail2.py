# -*- coding: utf-8 -*-
"""v0.8.5 收尾 v2：引号不可知的跨度替换。"""
import io
import re

BS = chr(92)


def span_edit(path, ops):
    s = io.open(path, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    for start, end, new in ops:
        i = s.find(start)
        assert i >= 0, "start: " + start[:24]
        j = s.find(end, i)
        assert j > i, "end: " + end[:24]
        j += len(end)
        s = s[:i] + new + s[j:]
        print("ok:", start[:20])
    io.open(path, "w", encoding="utf-8", newline="").write(s)


edit_zh = r"免训练内部信号不足以否决"
edit_zh_end = "的基座模型}"
new_zh = (r"部署教训因有层次而更冷峻：" + BS + "emph{免训练内部信号不足以否决"
          + '"自信地犯错"的基座模型}。')

edit_t = r"这一边界是方差型信号原理性盲区的推广：像素方差式 UQ 度量的是合成过程的方差"
edit_t_end = "而不是内容的临床显著性"
REF_TUMOR = "（图~" + BS + "ref{fig:tumor}）。"
new_t = (r"这一边界是方差型信号原理性盲区的推广：像素方差式 UQ 度量的是合成过程的方差"
         "而非内容的临床显著性，因此面向临床的 UQ 层必须对照病理标注审计（"
         + BS + "S" + BS + "ref{sec:discussion}，图~" + BS + "ref{fig:tumor}）。")

s = io.open(r"D:\MedTranslateProject\docs\latex\main_zh.tex",
            encoding="utf-8", newline="").read().replace("\r\n", "\n")

# 1) 崩塌教训句：从"部署教训因有层次"到"不是一道门。"整段重写（引号无关）
i = s.find("部署教训因有层次")
j = s.find("不是一道门。", i)
assert i >= 0 and j > i, (i, j)
j += len("不是一道门。")
s = s[:i] + new_zh + s[j:]
print("ok: slogan-zh")

# 2) 肿瘤教训句：若 v0.8.5 新句已存在则跳过；否则从"教训超出本文方法"到图引用结尾重写
if "方差型信号原理性盲区的推广" not in s:
    i = s.find("教训超出本文方法")
    j = s.find(REF_TUMOR, i)
    assert i >= 0 and j > i
    j += len(REF_TUMOR)
    s = s[:i] + new_t + s[j:]
    print("ok: tumor-lesson-zh")
else:
    print("skip: tumor-lesson already present")

# 3) 结论"收割"
s = s.replace("收割这些证据能为合成输出提供", "提取这些证据能为合成输出提供", 1)
io.open(r"D:\MedTranslateProject\docs\latex\main_zh.tex",
        "w", encoding="utf-8", newline="").write(s)

# related_work harvest
p = r"D:\MedTranslateProject\docs\latex\related_work.tex"
s = io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
s = s.replace("we keep the model frozen and harvest two signals that SelfRDB",
              "we keep the model frozen and extract two signals that SelfRDB", 1)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok: related_work harvest")
