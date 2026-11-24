"""Render manuscript/manuscript.md to a print-ready PDF (A4) with Chromium."""
import os, re, base64, markdown
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
NAME = sys.argv[1] if len(sys.argv) > 1 else "manuscript"
OUT = sys.argv[2] if len(sys.argv) > 2 else "Ibrahim_AIforFinance_Capstone_Manuscript.pdf"
src = open(os.path.join(ROOT, "manuscript", NAME + ".md")).read()

def embed(m):
    alt, path = m.group(1), m.group(2)
    p = os.path.normpath(os.path.join(ROOT, "manuscript", path))
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    num, _, rest = alt.partition(". ")
    return f'<figure><img src="data:image/png;base64,{b64}"><figcaption><b>{num}.</b> {rest}</figcaption></figure>'

src = re.sub(r"!\[(.*?)\]\((.*?)\)", embed, src)
title, _, body = src.partition("\n")
html_body = markdown.markdown(body, extensions=["tables"])
html_body = re.sub(r"<p><strong>(Table \d+\..*?)</strong></p>", r'<p class="tcap"><b>\1</b></p>', html_body)
def numcells(m):
    tag, attrs, content = m.group(1), m.group(2), m.group(3)
    txt = re.sub(r"<.*?>", "", content).strip()
    if txt and re.fullmatch(r"[\d.,%€+\-−–/ kWhEURMW()a-z]*\d[\d.,%€+\-−– /kWh]*", txt) and not re.search(r"[A-Za-z]{4,}", txt):
        return f'<{tag} class="n"{attrs}>{content}</{tag}>'
    return m.group(0)
html_body = re.sub(r"<(td|th)([^>]*)>(.*?)</\1>", numcells, html_body)
# references: one paragraph -> list
html_body = html_body.replace("<h2>References</h2>\n<p>", '<h2>References</h2>\n<p class="refs">')
css = """
@page { size: A4; }
body { font-family: 'Source Serif 4', 'Georgia', serif; font-size: 10.4pt; line-height: 1.45; color: #111; }
h1 { font-size: 17pt; line-height: 1.22; margin: 0 0 6pt; font-weight: 700; }
.author { font-size: 10pt; color: #444; margin-bottom: 14pt; border-bottom: 0.6pt solid #999; padding-bottom: 8pt; }
h2 { font-size: 12.2pt; margin: 16pt 0 5pt; border-bottom: 0.5pt solid #bbb; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 10.8pt; margin: 11pt 0 4pt; break-after: avoid; }
p { margin: 0 0 6.5pt; text-align: justify; hyphens: auto; }
ul, ol { margin: 0 0 6pt 16pt; padding: 0; } li { margin-bottom: 3pt; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 3pt 0 10pt; font-size: 8.9pt; break-inside: avoid; font-variant-numeric: tabular-nums; }
th { text-align: left; border-top: 1pt solid #111; border-bottom: 0.6pt solid #111; padding: 3pt 5pt; font-weight: 600; }
td { padding: 2.4pt 5pt; border-bottom: 0.3pt solid #ddd; } tr:last-child td { border-bottom: 1pt solid #111; }
td.n, th.n { text-align: right; } td:first-child, th:first-child { text-align: left !important; }
.tcap { margin: 8pt 0 2pt; font-size: 9.3pt; text-align: left; break-after: avoid; }
figure { margin: 8pt 0 10pt; break-inside: avoid; text-align: center; }
figure img { width: 88%; max-width: 14.2cm; }
figcaption { font-size: 8.8pt; color: #333; text-align: left; margin-top: 3pt; }
.abstract { background: #f4f4f1; padding: 8pt 10pt; border-left: 2pt solid #2a78d6; }
.refs { font-size: 9pt; text-align: left; white-space: pre-line; }
"""
head = title.lstrip("# ").strip()
first_p_end = html_body.index("</p>") + 4
author = html_body[:first_p_end].replace("<p>", '<p class="author">', 1)
rest = html_body[first_p_end:]
# abstract box
rest = re.sub(r"(<h2>Abstract</h2>\s*<p>)(.*?)(</p>)", r'<h2>Abstract</h2><p class="abstract">\2</p>', rest, count=1, flags=re.S)
html = f"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=swap" rel="stylesheet">
<style>{css}</style></head><body><h1>{head}</h1>{author}{rest}</body></html>"""
out_html = os.path.join(ROOT, "manuscript", NAME + ".html")
open(out_html, "w").write(html)
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.goto("file://" + out_html); pg.wait_for_timeout(500)
    pg.pdf(path=os.path.join(ROOT, "manuscript", OUT), format="A4",
           margin={"top": "20mm", "bottom": "18mm", "left": "19mm", "right": "19mm"},
           display_header_footer=True, header_template="<span></span>",
           footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#777"><span class="pageNumber"></span></div>')
    b.close()
print("pdf written")
