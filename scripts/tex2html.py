r"""main.tex → paper_preview.html 转换器（KaTeX 渲染公式）。

解决 main.tex 与 paper_pdf.py 内嵌 HTML 双份手工同步的问题：
paper_pdf.py 改为调用本模块，从 main.tex 单一来源生成预览。

支持子集：title/author/date、abstract、section/subsection/paragraph、
enumerate/itemize、table+tabular(booktabs/multicolumn)、equation、
\cite（按出现顺序编号，参考文献从 references.bib 生成）、\ref/\label、
\emph/\textbf/\texttt/\url、行内/展示数学（交给 KaTeX）。
"""
import re

LATEX_DIR = None  # 由调用方注入


def _read(path):
    import os
    return open(path, encoding="utf-8").read()


def _strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def _inline(s):
    s = s.replace("\\&", "&amp;").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = s.replace("\\&", "&")
    s = re.sub(r"\\emph\{([^{}]*)\}", r"<em>\1</em>", s)
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"<b>\1</b>", s)
    s = re.sub(r"\\texttt\{([^{}]*)\}", r"<code>\1</code>", s)
    s = re.sub(r"\\url\{([^{}]*)\}", r'<a href="\1">\1</a>', s)
    s = s.replace("\\TrustScore{}", "Trust Score").replace("\\TrustMap{}", "trust map")
    s = s.replace("\\cref{}", "Convergence Residual Field (CReF)")
    s = s.replace("---", "—").replace("--", "–")
    s = s.replace("``", "“").replace("''", "”")
    s = s.replace("$\\times$", "×").replace("$\\to$", "→").replace("$\\ge$", "≥")
    s = s.replace("\\S", "§").replace("\\,", " ").replace("~", " ")
    s = s.replace("\\\\", "<br>")
    return s


class Converter:
    def __init__(self, latex_dir):
        self.dir = latex_dir
        self.cite_order = []
        self.counters = {"table": 0, "section": 0, "subsection": 0}
        self.labels = {}

    def bib_map(self):
        bib = _read(self.dir + "/references.bib")
        entries = {}
        for m in re.finditer(r"@(\w+)\{([^,]+),", bib):
            entries[m.group(2).strip()] = m.group(1)
        return entries

    def ref_html(self):
        bib = _read(self.dir + "/references.bib")
        out, n = [], 0
        for key in self.cite_order:
            m = re.search(r"@\w+\{" + re.escape(key) + r",(.*?)\n\}", bib, re.S)
            n += 1
            if not m:
                out.append(f"<li>[{key}]</li>")
                continue
            body = m.group(1)

            def fld(name):
                mm = re.search(name + r"\s*=\s*[{\"](.*?)[}\"],?\s*\n", body, re.S)
                return re.sub(r"\s+", " ", mm.group(1)).strip("{} ") if mm else ""

            author = fld("author") or fld("editor")
            title = fld("title").rstrip(".")
            venue = fld("journal") or fld("booktitle") or fld("publisher")
            year = fld("year")
            a = author.split(" and ")[0].split(",")[0] if author else key
            etal = " et al." if " and " in author else ""
            out.append(f"<li>{a}{etal} ({year}). {title}. <i>{venue}</i>.</li>")
        return "<ol class=\"refs\">" + "".join(out) + "</ol>"


def tex_to_html(latex_dir):
    c = Converter(latex_dir)
    s = _strip_comments(_read(latex_dir + "/main.tex"))
    s = re.sub(r"\\input\{(\w+)\}", lambda m: _read(f"{latex_dir}/{m.group(1)}.tex"), s)
    s = re.sub(r"(?s)\\begin\{document\}(.*?)\\end\{document\}", r"\1", s)
    # 前言中的 title/author/date
    pre = s
    title = re.search(r"(?s)\\title\{(.*?)\}\s*\n\\author", pre)
    author = re.search(r"(?s)\\author\{(.*?)\}\s*\n\\date", pre)
    date = re.search(r"\\date\{([^{}]*)\}", pre)
    s = re.search(r"(?s)\\maketitle(.*)", s).group(1)

    def cite(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        out = []
        for k in keys:
            if k not in c.cite_order:
                c.cite_order.append(k)
            out.append(str(c.cite_order.index(k) + 1))
        return f"<sup>[{','.join(out)}]</sup>"

    # 依序处理块
    body = []
    pos = 0
    env_re = re.compile(r"(?s)\\begin\{(table|equation|abstract)\*?\}")
    list_re = re.compile(r"(?s)\\begin\{(enumerate|itemize)\}")
    sec_re = re.compile(r"\\(section|subsection|paragraph)\{((?:[^{}]|\{[^{}]*\})*)\}")

    def flush(t):
        # 行内 cite
        t = re.sub(r"~?\\cite\{([^{}]*)\}", cite, t)
        body.append(_inline(t))

    while pos < len(s):
        m_env = env_re.search(s, pos)
        m_list = list_re.search(s, pos)
        m_sec = sec_re.search(s, pos)
        cands = [x for x in (m_env, m_list, m_sec) if x]
        if not cands:
            flush(s[pos:])
            break
        m = min(cands, key=lambda x: x.start())
        flush(s[pos:m.start()])
        if m == m_sec:
            lvl, txt = m.group(1), m.group(2).strip()
            txt = re.sub(r"\\label\{[^{}]*\}", "", txt)
            if lvl == "section":
                c.counters["subsection"] = 0
                c.counters["section"] += 1
                num = c.counters["section"]
                body.append(f"<h2>{num}. {_inline(txt)}</h2>")
            elif lvl == "subsection":
                c.counters["subsection"] += 1
                body.append(f"<h3>{c.counters['section']}.{c.counters['subsection']} {_inline(txt)}</h3>")
            else:
                body.append(f"<h4>{_inline(txt)}</h4>")
            pos = m.end()
        elif m == m_env:
            env = m.group(1)
            end = s.index(r"\end{" + env + "}", m.end())
            content = s[m.end():end]
            pos = end + len(r"\end{" + env + "}")
            if env == "abstract":
                t = re.sub(r"\\cite\{([^{}]*)\}", cite, content)
                body.append(f'<div class="abstract"><b>ABSTRACT</b><br>{_inline(t)}</div>')
            elif env == "equation":
                eq = re.sub(r"\\label\{[^{}]*\}", "", content).strip()
                body.append(f"<div class='eq'>$${eq}$$</div>")
            else:  # table
                lab = re.search(r"\\label\{([^{}]*)\}", content)
                cap = re.search(r"(?s)\\caption\{(.*?)\}\s*\\label", content)
                c.counters["table"] += 1
                if lab:
                    c.labels[lab.group(1)] = c.counters["table"]
                rows_html = tabular_to_html(content, c)
                capt = re.sub(r"\\cite\{([^{}]*)\}", cite, cap.group(1)) if cap else ""
                body.append(f"<div class='tbl'><div class='cap'><b>Table {c.counters['table']}.</b> {_inline(capt)}</div>{rows_html}</div>")
        else:
            env = m.group(1)
            end = s.index(r"\end{" + env + "}", m.end())
            content = s[m.end():end]
            pos = end + len(r"\end{" + env + "}")
            items = [re.sub(r"\\item\s*", "", it) for it in re.split(r"\\item\s*", content) if it.strip()]
            tag = "ol" if env == "enumerate" else "ul"
            body.append(f"<{tag}>" + "".join(f"<li>{_inline(re.sub(r'\\cite\{([^{}]*)\}', cite, it))}</li>" for it in items) + f"</{tag}>")

    html = "".join(body)
    # \ref 解析
    html = re.sub(r"\\ref\{([^{}]*)\}", lambda m: str(c.labels.get(m.group(1), "?")), html)
    refs = c.ref_html()
    return title, author, date, html, refs


def tabular_to_html(content, c):
    m = re.search(r"(?s)\\begin\{tabular\}\{[^{}]*\}(.*?)\\end\{tabular\}", content)
    rows_raw = m.group(1)
    rows = []
    for line in re.split(r"\\\\", rows_raw):
        line = line.strip()
        if not line or re.fullmatch(r"(\\toprule|\\midrule|\\bottomrule|\\hline)", line.strip()):
            continue
        line = re.sub(r"\\(toprule|midrule|bottomrule|hline)", "", line).strip()
        if not line:
            continue
        cells = [x.strip() for x in line.split("&")]
        rows.append(cells)
    html = ["<table>"]
    for i, cells in enumerate(rows):
        is_head = i == 0
        tds = []
        for cell in cells:
            mm = re.match(r"\\multicolumn\{(\d+)\}\{\w+\}\{(.*)\}", cell)
            span, txt = (mm.group(1), mm.group(2)) if mm else ("1", cell)
            tag = "th" if is_head else "td"
            colspan = f" colspan='{span}'" if span != "1" else ""
            t = re.sub(r"\\cite\{([^{}]*)\}", lambda m2: f"<sup>[{m2.group(1)}]</sup>", txt)
            tds.append(f"<{tag}{colspan}>{_inline(t)}</{tag}>")
        html.append("<tr>" + "".join(tds) + "</tr>")
    html.append("</table>")
    return "".join(html)


if __name__ == "__main__":
    import os, sys
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "docs", "latex")
    title, author, date, html, refs = tex_to_html(d)
    print("OK, cite count:", len(Converter(d).cite_order) if False else "see file")
