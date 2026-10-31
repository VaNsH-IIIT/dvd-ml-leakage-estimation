"""Build the typed report: reports/REPORT.md -> reports/REPORT.html -> reports/REPORT.pdf.

Converts the Markdown subset the report uses (headings, paragraphs, nested
bullet and numbered lists, tables, bold/italic/code, images) to HTML with
print styling, then prints it to PDF with headless Chrome (no extra Python
packages needed).

Run:  python scripts/e1_build_report.py
"""
import html
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "reports" / "REPORT.md"
OUT_HTML = ROOT / "reports" / "REPORT.html"
OUT_PDF = ROOT / "reports" / "REPORT.pdf"
CHROME = "google-chrome"

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 10pt; line-height: 1.38; color: #111; }
h1 { font-size: 17pt; margin: 0 0 6pt; }
h2 { font-size: 13pt; margin: 16pt 0 6pt; border-bottom: 1px solid #bbb; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; break-after: avoid; }
p { margin: 5pt 0; }
ul, ol { margin: 3pt 0 5pt; padding-left: 18pt; }
li { margin: 1.5pt 0; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.6pt; background: #f2f2f0; padding: 0 2px; border-radius: 2px; }
table { border-collapse: collapse; margin: 6pt 0 8pt; font-size: 8.6pt; break-inside: avoid; }
th, td { border: 1px solid #c8c8c8; padding: 2pt 5pt; text-align: left; vertical-align: top; }
th { background: #f0f0ee; }
.figs { display: flex; gap: 6pt; justify-content: center; flex-wrap: wrap; margin: 6pt 0 2pt; break-inside: avoid; }
.figs img { max-width: 100%; }
.figs.n3 img { width: 32.5%; }
.figs.n1 img { width: 100%; }
.figs.n1.small img { width: 62%; }
.caption { font-size: 8.8pt; color: #444; text-align: center; margin: 2pt 0 10pt; }
"""

_IMG = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def inline(s):
    s = html.escape(s, quote=False).replace("\\|", "|")
    parts = re.split(r"(`[^`]+`)", s)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            out.append(f"<code>{part[1:-1]}</code>")
            continue
        part = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", part)
        part = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", part)
        out.append(part)
    return "".join(out)


def table(rows):
    cells = [[c.strip() for c in re.split(r"(?<!\\)\|", r.strip().strip("|"))] for r in rows]
    head, body = cells[0], cells[2:]
    h = "".join(f"<th>{inline(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def figures(line):
    imgs = _IMG.findall(line)
    small = len(imgs) == 1 and ("parity_parity" in imgs[0][1] or "error_vs" in imgs[0][1])
    cls = f"figs n{min(len(imgs), 3)}" + (" small" if small else "")
    return f'<div class="{cls}">' + "".join(f'<img src="{src}" alt="{html.escape(alt)}">' for alt, src in imgs) + "</div>"


_ITEM = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")


def convert(md):
    lines = md.splitlines()
    out, i = [], 0
    stack = []   # open lists: (indent, tag)

    def close_lists(to_indent=-1):
        while stack and stack[-1][0] > to_indent:
            out.append(f"</li></{stack.pop()[1]}>")

    while i < len(lines):
        line = lines[i]
        m = _ITEM.match(line)
        if m:
            indent, marker, text = len(m.group(1)), m.group(2), m.group(3)
            tag = "ol" if marker[0].isdigit() else "ul"
            while stack and stack[-1][0] > indent:
                out.append(f"</li></{stack.pop()[1]}>")
            if stack and stack[-1][0] == indent:
                out.append(f"</li><li>{inline(text)}")
            else:
                out.append(f"<{tag}><li>{inline(text)}")
                stack.append((indent, tag))
            i += 1
            continue
        if stack and line.strip() and line.startswith(" ") and not line.strip().startswith("|"):
            out.append(" " + inline(line.strip()))   # continuation of a list item
            i += 1
            continue
        if not line.strip():
            # a blank line ends lists unless the next line continues one
            nxt = next((l for l in lines[i + 1:] if l.strip()), "")
            if not _ITEM.match(nxt):
                close_lists()
            i += 1
            continue
        close_lists()
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
        elif line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(table(rows))
            continue
        elif _IMG.search(line):
            out.append(figures(line))
        elif line.startswith("*") and line.rstrip().endswith("*") and not line.startswith("**"):
            out.append(f'<p class="caption">{inline(line.strip()[1:-1])}</p>')
        else:
            para = [line.strip()]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||!\[|\s*([-*]|\d+\.)\s)", lines[i + 1]):
                i += 1
                para.append(lines[i].strip())
            out.append(f"<p>{inline(' '.join(para))}</p>")
        i += 1
    close_lists()
    return "\n".join(out)


def main():
    body = convert(SRC.read_text())
    OUT_HTML.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>Team 22 report</title>'
                        f"<style>{CSS}</style></head><body>{body}</body></html>")
    OUT_PDF.unlink(missing_ok=True)   # so a failed print can't leave an old PDF looking fresh
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not OUT_PDF.exists():
        sys.exit(f"PDF not written: {res.stderr[-500:]}")
    print(f"wrote {OUT_PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
