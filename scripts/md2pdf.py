#!/usr/bin/env python3
"""Converts Markdown → PDF via WeasyPrint (full CSS support: tables, RTL Arabic, code) — auto-detects Arabic.

Usage: tools/rl-venv/bin/python scripts/md2pdf.py <out_dir> <file1.md> [file2.md ...]
"""
import os
import re
import sys

import markdown
from weasyprint import HTML

EMOJI = '"Noto Color Emoji"'
CSS = """
@page { margin: 2cm 1.6cm; }
h1 { font-size: 18pt; }
h2 { font-size: 14pt; border-bottom: 1px solid #ccc; padding-bottom: 2px; margin-top: 14px; }
h3 { font-size: 12pt; } h4 { font-size: 11pt; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; }
td, th { border: 1px solid #999; padding: 4px 7px; vertical-align: top; }
th { background: #f0f0f0; }
code { background: #f3f3f3; padding: 1px 3px; direction: ltr; unicode-bidi: embed;
       font-family: "DejaVu Sans Mono", monospace; font-size: 9pt; }
pre { background: #f6f6f6; padding: 8px; border: 1px solid #ddd; direction: ltr;
      text-align: left; white-space: pre-wrap; }
pre code { background: none; }
blockquote { border-inline-start: 3px solid #bbb; margin: 6px 0; padding: 2px 12px; color: #444; }
img { max-width: 100%; }
"""


def is_arabic(text):
    ar = len(re.findall(r"[\u0600-\u06FF]", text))
    la = len(re.findall(r"[A-Za-z]", text))
    return ar > la * 0.6


def convert(md_path, out_dir):
    md = open(md_path, encoding="utf-8").read()
    rtl = is_arabic(md)
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    if rtl:
        base_css = (f'body{{font-family:"Noto Sans Arabic","Noto Naskh Arabic","DejaVu Sans",{EMOJI};'
                    f'font-size:11pt;line-height:1.6;direction:rtl;text-align:right;}}')
        attr = ' dir="rtl" lang="ar"'
    else:
        base_css = f'body{{font-family:"Noto Sans","DejaVu Sans",{EMOJI};font-size:11pt;line-height:1.45;}}'
        attr = ' lang="en"'
    doc = (f'<!DOCTYPE html><html{attr}><head><meta charset="utf-8">'
           f"<style>{base_css}{CSS}</style></head><body>{body}</body></html>")
    base = os.path.splitext(os.path.basename(md_path))[0]
    out = os.path.join(out_dir, base + ".pdf")
    HTML(string=doc, base_url=os.path.dirname(md_path)).write_pdf(out)
    print(f"  {'AR' if rtl else 'EN'}  {base}.pdf")


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    for md in sys.argv[2:]:
        try:
            convert(md, out_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {os.path.basename(md)}: {e}")


if __name__ == "__main__":
    main()
