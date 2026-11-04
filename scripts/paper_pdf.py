"""论文初稿预览：main.tex / main_zh.tex → HTML（tex2html 转换）→ PDF（Edge 渲染）。

初稿审阅用。正式投稿版在 Overleaf 用 main.tex + references.bib 编译。
用法:
  python scripts/paper_pdf.py                # 英文版 main.tex → TrustBridge_paper_draft.pdf
  python scripts/paper_pdf.py --zh           # 中文版 main_zh.tex → TrustBridge_paper_draft_zh.pdf
"""
import argparse
import os

from playwright.sync_api import sync_playwright

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from tex2html import tex_to_html  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--zh", action="store_true", help="转换中文版 main_zh.tex")
args = ap.parse_args()

LATEX = os.path.join(ROOT, "docs", "latex")
SRC_NAME = "main_zh.tex" if args.zh else "main.tex"
OUT = os.path.join(LATEX, "TrustBridge_paper_draft_zh.pdf" if args.zh
                   else "TrustBridge_paper_draft.pdf")
SRC = os.path.join(LATEX, "paper_preview_zh.html" if args.zh else "paper_preview.html")

title, author, date, html, refs = tex_to_html(LATEX, src=SRC_NAME)
FONT_STACK = ("'Microsoft YaHei','PingFang SC','Noto Sans SC',"
              "'Georgia','Times New Roman',serif" if args.zh
              else "'Georgia','Times New Roman',serif")


title_h = "<h1>" + title + "</h1>" if title else "<h1>TrustBridge</h1>"
author_h = "<div class='authors'>" + author + "</div>" if author else ""
date_h = f"<div class='affil'>{date}</div>" if date else ""

HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>TrustBridge Paper Draft v0.2</title>
<link rel="stylesheet" href="https://registry.npmmirror.com/katex/0.16.9/files/dist/katex.min.css">
<script defer src="https://registry.npmmirror.com/katex/0.16.9/files/dist/katex.min.js"></script>
<script defer src="https://registry.npmmirror.com/katex/0.16.9/files/dist/contrib/auto-render.min.js"
 onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}],throwOnError:false}});"></script>
<style>
 body{{font-family:{FONT_STACK};max-width:860px;margin:0 auto;
      padding:40px 50px;color:#111;line-height:1.72;font-size:15px;background:#fff;}}
 h1{{font-size:24px;text-align:center;margin-bottom:4px;}}
 h2{{font-size:17px;margin:26px 0 8px;border-bottom:1.5px solid #2563eb;padding-bottom:3px;color:#1e3a8a;}}
 h3{{font-size:15px;margin:16px 0 6px;color:#1e3a8a;}}
 h4{{font-size:14px;margin:14px 0 4px;color:#334155;}}
 .subtitle{{font-size:14px;font-weight:400;color:#475569;margin-top:6px;}}
 .authors{{text-align:center;font-size:14px;color:#444;margin-bottom:6px;}}
 .affil{{text-align:center;font-size:12px;color:#666;margin-bottom:4px;}}
 .abstract{{margin:20px 40px;font-size:13.5px;background:#f8fafc;padding:14px 18px;border-radius:6px;}}
 .abstract b{{font-size:12px;letter-spacing:1px;}}
 table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:12.5px;}}
 th,td{{border:1px solid #ccc;padding:4px 7px;text-align:center;}}
 th{{background:#eff6ff;}}
 .tbl{{margin:14px 0;}}
 .cap{{font-size:12.5px;color:#555;margin-bottom:4px;line-height:1.5;}}
 .eq{{margin:12px 0;text-align:center;}}
 .refs{{font-size:12.5px;line-height:1.6;}}
 .draft-banner{{background:#fef3c7;border:1px solid #f59e0b;border-radius:6px;
               padding:8px 16px;font-size:12.5px;color:#92400e;margin-bottom:24px;text-align:center;}}
 sup{{font-size:11px;}}
 ol{{padding-left:22px;}} li{{margin:4px 0;}}
</style></head><body>
<div class="draft-banner">DRAFT v0.4 (auto-converted from {SRC_NAME}) — internal review ·
compile source: main.tex + references.bib + related_work.tex (Overleaf-ready)</div>
{title_h}{author_h}{date_h}
{html}
<h2>References</h2>
{refs}
</body></html>"""

with open(SRC, "w", encoding="utf-8") as f:
    f.write(HTML)

with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge")
    page = b.new_page(viewport={"width": 1000, "height": 800})
    page.goto("file:///" + SRC.replace("\\", "/"))
    page.wait_for_timeout(4000)
    page.pdf(path=OUT, format="A4", print_background=True,
             margin={"top": "12mm", "bottom": "12mm", "left": "10mm", "right": "10mm"})
    b.close()
print("PDF ->", OUT, os.path.getsize(OUT) // 1024, "KB")
