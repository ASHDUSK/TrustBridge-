r"""main.tex → paper_preview.html 转换器 v3。

管线（每段正文 _flush）：
  1. _dita：LaTeX 重音/微转义 → Unicode（作用于原文）
  2. _esc：HTML 转义（& < >）
  3. resolve \ref{label} → 编号（查预注册的 labels）
  4. cite_pass：\cite{key} → <sup>[n]</sup>（转义后生成，不再被逃逸）
  5. _cmds：命令替换（emph/textbf/url/TrustMap/cref/破折号/引号/\S 等）
  6. 数学保护：$...$ 区间跳过命令替换与花括号清理，交给 KaTeX
  7. 非数学区：剥除残余成对花括号
"""
import re


def _read(path):
    return open(path, encoding="utf-8").read()


_ACCENTS = [(r'{\"o}', "ö"), (r'{\"a}', "ä"), (r'{\"u}', "ü"),
            (r'\"{o}', "ö"), (r'\"{a}', "ä"), (r'\"{u}', "ü"),
            (r'\"o', "ö"), (r'\"a', "ä"), (r'\"u', "ü"), (r'\"O', "Ö"),
            (r'\"A', "Ä"), (r'\"U', "Ü"), (r"\'e", "é"), (r"\'E", "É"),
            (r"\`e", "è"), (r"\'a", "á")]


def _dita(s):
    for a, b in _ACCENTS:
        s = s.replace(a, b)
    s = s.replace("{,}", ",")
    s = re.sub(r"\\\.(\s)", r".\1", s)
    return s


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Converter:
    def __init__(self):
        self.cite_order = []
        self.counters = {"table": 0, "section": 0, "subsection": 0}
        self.labels = {}

    def resolve_ref(self, m):
        key = m.group(1)
        if key.startswith("sec:"):
            return "§" + str(self.labels.get(key, key))
        return str(self.labels.get(key, key))

    def cite_pass(self, s):
        def rep(m):
            keys = [k.strip() for k in m.group(1).split(",")]
            out = []
            for k in keys:
                if k not in self.cite_order:
                    self.cite_order.append(k)
                out.append(str(self.cite_order.index(k) + 1))
            return "<sup>[" + ",".join(out) + "]</sup>"
        return re.sub(r"~?\\cite\{([^{}]*)\}", rep, s)


def _cmds(c, s):
    s = re.sub(r"\\emph\{([^{}]*)\}", r"<em>\1</em>", s)
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"<b>\1</b>", s)
    s = re.sub(r"\\texttt\{([^{}]*)\}", r"<code>\1</code>", s)
    s = re.sub(r"\\url\{([^{}]*)\}", r'<a href="\1">\1</a>', s)
    s = s.replace("\\TrustMap", "trust map").replace("\\TrustScore", "Trust Score")
    s = s.replace("\\cref", "Convergence Residual Field (CReF)")
    s = s.replace("---", "—").replace("--", "–")
    s = s.replace("``", "“").replace("''", "”")
    s = s.replace("\\%", "%").replace("\\ ", " ").replace("vs\\.", "vs.")
    s = s.replace("$\\times$", "×").replace("$\\to$", "→").replace("$\\ge$", "≥")
    s = s.replace("\\S", "§").replace("\\,", " ").replace("~", " ")
    s = s.replace("\\\\", "<br>")
    s = re.sub(r"\\item\s*", "", s)
    # 数学保护：$...$ 之外剥除残余成对花括号与裸命令名
    parts = re.split(r"(\$[^$]*\$)", s)
    for i, p in enumerate(parts):
        if i % 2 == 0:
            p = re.sub(r"\{\}?", "", p)
            p = re.sub(r"\\(emph|textbf|texttt|url|large|small|centering)\b", "", p)
            parts[i] = p
    return "".join(parts)


def _flush(c, t):
    t = _dita(t)
    t = _esc(t)
    t = re.sub(r"\\ref\{([^{}]*)\}", c.resolve_ref, t)
    t = c.cite_pass(t)
    return _cmds(c, t)


def tex_to_html(latex_dir, src="main.tex"):
    c = Converter()
    s = _read(latex_dir + "/" + src)
    s = re.sub(r"(?m)^%.*$", "", s)
    s = re.sub(r"\\input\{(\w+)\}", lambda m: _read(f"{latex_dir}/{m.group(1)}.tex"), s)
    m = re.search(r"\\bibliographystyle", s)
    if m:
        s = s[:m.start()]
    s = re.sub(r"(?s)\\begin\{document\}(.*?)\\end\{document\}", r"\1", s)

    title = re.search(r"(?s)\\title\{(.*?)\}\s*\\author", s)
    author = re.search(r"(?s)\\author\{(.*?)\}\s*\\date", s)
    date = re.search(r"\\date\{([^{}]*)\}", s)
    s = s[s.index("\\maketitle") + len("\\maketitle"):]

    # ---- 预注册 section / table / equation 的 label ----
    sec_re = re.compile(r"\\(section|subsection)\{((?:[^{}]|\{[^{}]*\})*)\}")
    cnt = {"section": 0, "subsection": 0}
    for m in sec_re.finditer(s):
        if m.group(1) == "section":
            cnt["subsection"] = 0
            cnt["section"] += 1
            num = str(cnt["section"])
        else:
            cnt["subsection"] += 1
            num = f"{cnt['section']}.{cnt['subsection']}"
        tail = s[m.end():m.end() + 60]
        lm = re.match(r"\s*\\label\{([^{}]*)\}", tail)
        if lm:
            c.labels[lm.group(1)] = num
    for i, m in enumerate(re.finditer(r"(?s)\\begin\{table\}.*?\\label\{([^{}]*)\}", s)):
        c.labels[m.group(1)] = str(i + 1)
    for i, m in enumerate(re.finditer(r"(?s)\\begin\{equation\}.*?\\label\{([^{}]*)\}", s)):
        c.labels[m.group(1)] = str(i + 1)
    s = re.sub(r"\\label\{[^{}]*\}", "", s)

    body = []
    pos = 0
    env_re = re.compile(r"(?s)\\begin\{(table|equation|abstract)\*?\}")
    list_re = re.compile(r"(?s)\\begin\{(enumerate|itemize)\}")
    sec_re2 = re.compile(r"\\(section|subsection|paragraph)\{((?:[^{}]|\{[^{}]*\})*)\}")

    while pos < len(s):
        m_env = env_re.search(s, pos)
        m_list = list_re.search(s, pos)
        m_sec = sec_re2.search(s, pos)
        cands = [x for x in (m_env, m_list, m_sec) if x]
        if not cands:
            body.append(_flush(c, s[pos:]))
            break
        m = min(cands, key=lambda x: x.start())
        body.append(_flush(c, s[pos:m.start()]))
        if m == m_sec:
            lvl, txt = m.group(1), m.group(2).strip()
            if lvl == "section":
                c.counters["subsection"] = 0
                c.counters["section"] += 1
                body.append(f"<h2>{c.counters['section']}. {_flush(c, txt)}</h2>")
            elif lvl == "subsection":
                c.counters["subsection"] += 1
                body.append(f"<h3>{c.counters['section']}.{c.counters['subsection']} {_flush(c, txt)}</h3>")
            else:
                body.append(f"<h4>{_flush(c, txt)}</h4>")
            pos = m.end()
        elif m == m_env:
            env = m.group(1)
            end = s.index(r"\end{" + env + "}", m.end())
            content = s[m.end():end]
            pos = end + len(r"\end{" + env + "}")
            if env == "abstract":
                body.append(f'<div class="abstract"><b>ABSTRACT</b><br>{_flush(c, content)}</div>')
            elif env == "equation":
                body.append(f"<div class='eq'>$${content.strip()}$$</div>")
            else:
                cap = re.search(r"(?s)\\caption\{(.*?)\}\s*(?:\\label|\\begin\{tabular\})", content)
                c.counters["table"] += 1
                rows_html = tabular_to_html(c, content)
                capt = _flush(c, cap.group(1)) if cap else ""
                body.append(f"<div class='tbl'><div class='cap'><b>Table {c.counters['table']}.</b> {capt}</div>{rows_html}</div>")
        else:
            env = m.group(1)
            end = s.index(r"\end{" + env + "}", m.end())
            content = s[m.end():end]
            pos = end + len(r"\end{" + env + "}")
            items = [re.sub(r"\\item\s*", "", it) for it in re.split(r"\\item\s*", content) if it.strip()]
            tag = "ol" if env == "enumerate" else "ul"
            body.append(f"<{tag}>" + "".join(f"<li>{_flush(c, it)}</li>" for it in items) + f"</{tag}>")

    html = "".join(body)
    refs = ref_html(latex_dir, c.cite_order)

    t = _clean_title(title.group(1)) if title else "TrustBridge"
    a = _clean_author(author.group(1)) if author else ""
    d = _flush(c, date.group(1)) if date else ""
    return t, a, d, html, refs


def _clean_title(t):
    t = re.sub(r"%[^\n]*", "", t)
    t = t.replace("\\\\", "\n")
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    main = _flush(Converter(), lines[0]) if lines else "TrustBridge"
    sub = ""
    for ln in lines[1:]:
        ln = re.sub(r"\\large\b", "", ln).strip()
        if ln:
            sub += ln + " "
    sub = _flush(Converter(), sub.strip())
    return f"{main}<div class='subtitle'>{sub}</div>" if sub else main


def _clean_author(a):
    a = re.sub(r"%[^\n]*", "", a)
    a = _dita(a)
    a = _esc(a)
    a = re.sub(r"\\small\b", "", a)
    a = re.sub(r'\\url\{([^{}]*)\}', r"<a href='\1'>\1</a>", a)
    a = a.replace("\\\\", "<br>")
    a = re.sub(r"\{\}?", "", a)
    return a


def _bib_field(body, name):
    m = re.search(name + r"\s*=\s*[{\"](.*?)[}\"],?\s*\n", body, re.S)
    if not m:
        return ""
    t = re.sub(r"\s+", " ", m.group(1)).strip()
    t = re.sub(r"[{}]", "", t)
    t = _dita(t)
    return t.strip().strip(".,; ")


def ref_html(latex_dir, cite_order):
    bib = _read(latex_dir + "/references.bib")
    out = []
    for i, key in enumerate(cite_order):
        m = re.search(r"@\w+\{" + re.escape(key) + r",(.*?)\n\}", bib, re.S)
        if not m:
            out.append(f"<li>[{key}]</li>")
            continue
        body = m.group(1)
        author = _bib_field(body, "author") or _bib_field(body, "editor")
        title = _bib_field(body, "title")
        venue = _bib_field(body, "journal") or _bib_field(body, "booktitle") or _bib_field(body, "publisher")
        year = _bib_field(body, "year")
        a = author.split(" and ")[0].split(",")[0] if author else key
        etal = " et al." if " and " in author else ""
        title = title.replace("--", "–")
        dot = "" if title.endswith(("?", "!")) else "."
        seg = f"{a}{etal} ({year}). {title}{dot}"
        if venue:
            seg += f" <i>{venue}</i>."
        out.append(f"<li>{seg}</li>")
    return "<ol class=\"refs\">" + "".join(out) + "</ol>"


def tabular_to_html(c, content):
    m = re.search(r"(?s)\\begin\{tabular\}\{[^{}]*\}(.*?)\\end\{tabular\}", content)
    rows = []
    for line in re.split(r"\\\\", m.group(1)):
        line = re.sub(r"\\(toprule|midrule|bottomrule|hline|cmidrule\(?lr?\)?\{?[^{}]*\}?)", "",
                      line).strip()
        if not line:
            continue
        rows.append([x.strip() for x in line.split("&")])
    html = ["<table>"]
    for i, cells in enumerate(rows):
        tds = []
        for cell in cells:
            mm = re.match(r"\\multicolumn\{(\d+)\}\{\w+\}\{(.*)\}", cell)
            span, txt = (mm.group(1), mm.group(2)) if mm else ("1", cell)
            span = span if span.isdigit() else "1"
            tag = "th" if i == 0 else "td"
            colspan = f" colspan='{span}'" if span != "1" else ""
            tds.append(f"<{tag}{colspan}>{_flush(c, txt)}</{tag}>")
        html.append("<tr>" + "".join(tds) + "</tr>")
    html.append("</table>")
    return "".join(html)


if __name__ == "__main__":
    import os
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "docs", "latex")
    t, a, dd, html, refs = tex_to_html(d)
    print("converted; sup tags:", html.count("<sup>"), "| leak check:",
          html.count("\\ref{") + html.count("\\label{") + html.count("<sup>["))
