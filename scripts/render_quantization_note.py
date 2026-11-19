"""Render the complete Markdown engineering note to a review PDF."""

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

root = Path(__file__).resolve().parents[1]
pdfmetrics.registerFont(TTFont("DVSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVBold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVMono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))
styles = {
    "body": ParagraphStyle("body", fontName="DVSans", fontSize=9, leading=13, spaceAfter=8),
    "title": ParagraphStyle("title", fontName="DVBold", fontSize=18, leading=24, spaceAfter=12),
    "heading": ParagraphStyle(
        "heading", fontName="DVBold", fontSize=12, leading=17, spaceBefore=10, spaceAfter=8
    ),
    "table": ParagraphStyle("table", fontName="DVSans", fontSize=8, leading=11),
    "code": ParagraphStyle("code", fontName="DVMono", fontSize=7.3, leading=11),
}


def inline(text):
    text = html.escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#1958ac">\1</link>', text)
    return re.sub(r"`([^`]+)`", r'<font name="DVMono">\1</font>', text)


lines = (root / "docs/ISOLATED_QUANTIZATION_MEASUREMENTS.md").read_text().splitlines()
story = []
i = 0
while i < len(lines):
    line = lines[i]
    if not line.strip():
        i += 1
        continue
    if line.startswith("# "):
        story.append(Paragraph(inline(line[2:]), styles["title"]))
        i += 1
    elif line.startswith("## "):
        story.append(Paragraph(inline(line[3:]), styles["heading"]))
        i += 1
    elif line.startswith("```"):
        block = []
        i += 1
        while i < len(lines) and not lines[i].startswith("```"):
            block.append(lines[i])
            i += 1
        story.append(Preformatted("\n".join(block), styles["code"]))
        i += 1
    elif line.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            if not re.fullmatch(r"[| :\-]+", lines[i]):
                rows.append(
                    [
                        Paragraph(inline(cell.strip()), styles["table"])
                        for cell in lines[i].strip("|").split("|")
                    ]
                )
            i += 1
        table = Table(rows, colWidths=[234, 98, 98, 98], repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7edf7")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d4dce8")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f8fafc")],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.extend([table, Spacer(1, 10)])
    else:
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "```")):
            paragraph.append(lines[i])
            i += 1
        story.append(Paragraph(inline(" ".join(paragraph)), styles["body"]))

output = root / "docs/ISOLATED_QUANTIZATION_MEASUREMENTS.pdf"
doc = SimpleDocTemplate(
    str(output),
    pagesize=letter,
    leftMargin=42,
    rightMargin=42,
    topMargin=42,
    bottomMargin=42,
    title="Quantized storage and loaded model memory",
    author="Olympus project",
)


def footer(canvas, doc):
    canvas.setFont("DVSans", 8)
    canvas.setFillColor(colors.HexColor("#4d6078"))
    canvas.drawString(42, 24, "Olympus engineering measurements / 7 October 2026")
    canvas.drawRightString(570, 24, str(doc.page))


doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(output)
