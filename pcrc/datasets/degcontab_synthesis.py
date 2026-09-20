"""DegConTab source construction (Appendix A.1.3-A.1.4).

The source is generated at the document level. For each document a topic
and table-structure templates with controlled header hierarchy,
ruling-line / border style, and visual-degradation settings are chosen; a
large language model then generates a multi-page document for that topic,
with its HTML tables embedded in the running text. The renderer converts
the generated document into page images and applies the requested visual
degradation (watermarks, stains, stamps); the artefact boxes it draws are
kept so that the text layer under them can be masked
(``pcrc.preprocessing.parsing.mask_degraded_text``), and degraded regions
are observed only through the image.

Controlled attribute shares (Table 5): header hierarchy L1-L5
25/35/20/12/8%; visual degradation none/watermark/stain/stamp 70/10/10/10%;
border style full/three-line/header-only/other 30/25/15/30%.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Dict, List, Optional, Sequence, Tuple

SYNTHESIS_SYSTEM = "You are generating source content for a controlled document table benchmark."

SYNTHESIS_PROMPT = """[Input]
The target document topic is: {DOCUMENT_TOPIC}.

The document must use these table-structure templates: {TABLE_STRUCTURE_TEMPLATES}.

The tables must follow these controlled difficulty settings:
- header hierarchy level: {HEADER_LEVEL};
- visual degradation type: {VISUAL_DEGRADATION};
- ruling-line / border style: {BORDER_STYLE}.

[Goal]
Generate a realistic multi-page document on the target topic. The document must contain several tables and surrounding paragraphs. Each table should follow one of the supplied structure templates, match the requested structural and visual settings, and be coherent with the surrounding text.

[Instructions]
1. Generate a document title and multiple ordered pages.
2. Place one or more HTML tables on each page when appropriate.
3. Write the surrounding paragraphs according to the natural relationship between text and tables in real documents of this kind; there is no need to deliberately avoid table-related content, nor to deliberately describe the tables.
4. Make each table structure match its assigned template and the requested header hierarchy level.
5. Make each table compatible with the requested ruling-line / border style, but do not write visual-degradation labels into table content.
6. Treat visual degradation as rendering metadata; it will be applied by the renderer.
7. Return only valid JSON.

[Output Schema]
{{
  "document_title": "...",
  "pages": [
    {{
      "page_index": 1,
      "paragraphs": ["...", "..."],
      "tables": [
        {{"template_id": "...", "table_html": "<table>...</table>"}}
      ]
    }}
  ],
  "render_controls": {{
    "header_level": "{HEADER_LEVEL}",
    "degradation": "{VISUAL_DEGRADATION}",
    "border": "{BORDER_STYLE}"
  }}
}}"""

HEADER_LEVELS = {"L1": 0.25, "L2": 0.35, "L3": 0.20, "L4": 0.12, "L5": 0.08}
DEGRADATIONS = {"none": 0.70, "watermark": 0.10, "stain": 0.10, "stamp": 0.10}
BORDER_STYLES = {"full": 0.30, "three-line": 0.25, "header-only": 0.15, "other": 0.30}


@dataclass
class RenderControls:
    header_level: str
    degradation: str
    border: str


def sample_controls(rng: random.Random) -> RenderControls:
    pick = lambda d: rng.choices(list(d), weights=list(d.values()))[0]  # noqa: E731
    return RenderControls(pick(HEADER_LEVELS), pick(DEGRADATIONS), pick(BORDER_STYLES))


def build_prompt(topic: str, templates: Sequence[str], controls: RenderControls) -> List[dict]:
    user = SYNTHESIS_PROMPT.format(DOCUMENT_TOPIC=topic, TABLE_STRUCTURE_TEMPLATES=", ".join(templates),
                                   HEADER_LEVEL=controls.header_level, VISUAL_DEGRADATION=controls.degradation,
                                   BORDER_STYLE=controls.border)
    return [{"role": "system", "content": SYNTHESIS_SYSTEM}, {"role": "user", "content": user}]


def generate_document(client, model: str, topic: str, templates: Sequence[str], controls: RenderControls) -> dict:
    resp = client.chat.completions.create(model=model, temperature=0.8, messages=build_prompt(topic, templates, controls))
    raw = resp.choices[0].message.content or ""
    return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])


# ------------------------------------------------------------ HTML -> grid
@dataclass
class Cell:
    row: int
    col: int
    rowspan: int
    colspan: int
    text: str
    header: bool


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: List[List[dict]] = []
        self._cell: Optional[dict] = None
        self._in_thead = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "thead":
            self._in_thead = True
        elif tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            self._cell = {"rowspan": int(a.get("rowspan", 1) or 1), "colspan": int(a.get("colspan", 1) or 1),
                          "text": "", "header": tag == "th" or self._in_thead}
            if not self.rows:
                self.rows.append([])
            self.rows[-1].append(self._cell)

    def handle_endtag(self, tag):
        if tag == "thead":
            self._in_thead = False
        elif tag in ("td", "th"):
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell["text"] += data


def html_to_grid(html: str) -> Tuple[List[Cell], int, int]:
    """Place the cells of an HTML table on a row x column grid, honouring spans."""
    parser = _TableParser()
    parser.feed(html)
    occupied = set()
    cells: List[Cell] = []
    n_cols = 0
    for r, row in enumerate(parser.rows):
        c = 0
        for cell in row:
            while (r, c) in occupied:
                c += 1
            for dr in range(cell["rowspan"]):
                for dc in range(cell["colspan"]):
                    occupied.add((r + dr, c + dc))
            cells.append(Cell(r, c, cell["rowspan"], cell["colspan"], " ".join(cell["text"].split()), cell["header"]))
            c += cell["colspan"]
            n_cols = max(n_cols, c)
    return cells, len(parser.rows), n_cols


# ---------------------------------------------------------------- layout
@dataclass
class RenderedPage:
    page_index: int
    image_path: str
    size: Tuple[float, float]
    blocks: List[dict]                      # {type, bbox, text | (table_id, html, cells)}
    words: List[dict]                       # text layer: {bbox, text, block} in page points
    artifacts: List[List[float]] = field(default_factory=list)   # boxes of injected degradation


FONT_SIZE = 10.0
LINE_H = 13.0
CELL_PAD = 3.0
MARGIN = 60.0


def _font(size: float, scale: float):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, int(size * scale))
        except OSError:
            continue
    return ImageFont.load_default()


def _text_width(font, text: str, scale: float) -> float:
    return font.getlength(text) / scale if hasattr(font, "getlength") else 0.55 * FONT_SIZE * len(text)


def _wrap(font, text: str, width: float, scale: float) -> List[str]:
    lines, cur = [], ""
    for w in text.split():
        cand = f"{cur} {w}".strip()
        if cur and _text_width(font, cand, scale) > width:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    return lines + [cur] if cur else lines


def _layout_paragraph(text: str, x0: float, y: float, width: float, font, scale: float, block_ref: int) -> Tuple[dict, List[dict]]:
    """Word-wrap one paragraph; return its block and one word box per word."""
    words, yy = [], y
    for line in _wrap(font, text, width, scale):
        x = x0
        for w in line.split():
            ww = _text_width(font, w, scale)
            words.append({"bbox": [x, yy, x + ww, yy + LINE_H], "text": w, "block": block_ref})
            x += ww + _text_width(font, " ", scale)
        yy += LINE_H
    return {"type": "paragraph", "bbox": [x0, y, x0 + width, yy], "text": text}, words


def _layout_table(html: str, table_id: str, x0: float, y: float, width: float, font, scale: float,
                  block_ref: int) -> Tuple[dict, List[dict]]:
    """Grid layout of one table: equal column widths, row height from the
    longest wrapped cell; cell boxes are kept for drawing and cropping."""
    cells, n_rows, n_cols = html_to_grid(html)
    if n_rows == 0 or n_cols == 0:
        return {"type": "table", "bbox": [x0, y, x0 + width, y + LINE_H], "table_id": table_id, "html": html, "cells": []}, []
    col_w = width / n_cols
    lines_of: Dict[int, List[str]] = {}
    row_h = [LINE_H + 2 * CELL_PAD] * n_rows
    for i, c in enumerate(cells):
        lines_of[i] = _wrap(font, c.text, c.colspan * col_w - 2 * CELL_PAD, scale) or [""]
        if c.rowspan == 1:
            row_h[c.row] = max(row_h[c.row], len(lines_of[i]) * LINE_H + 2 * CELL_PAD)
    row_y = [y]
    for h in row_h:
        row_y.append(row_y[-1] + h)
    words, boxes = [], []
    for i, c in enumerate(cells):
        bx0, by0 = x0 + c.col * col_w, row_y[c.row]
        bx1, by1 = bx0 + c.colspan * col_w, row_y[min(c.row + c.rowspan, n_rows)]
        boxes.append({"bbox": [bx0, by0, bx1, by1], "text": c.text, "header": c.header, "lines": lines_of[i]})
        yy = by0 + CELL_PAD
        for line in lines_of[i]:
            x = bx0 + CELL_PAD
            for w in line.split():
                ww = _text_width(font, w, scale)
                words.append({"bbox": [x, yy, x + ww, yy + LINE_H], "text": w, "block": block_ref})
                x += ww + _text_width(font, " ", scale)
            yy += LINE_H
    block = {"type": "table", "bbox": [x0, y, x0 + width, row_y[-1]], "table_id": table_id, "html": html,
             "cells": boxes, "header_rows": max([c.row + c.rowspan for c in cells if c.header], default=0),
             "row_y": row_y}
    return block, words


# ------------------------------------------------------------- artefacts
def _page_artifacts(rng: random.Random, degradation: str, page_size, n: int) -> List[List[float]]:
    """Page-level artefact boxes: watermarks span the page diagonal, stains
    and stamps are placed anywhere on the page."""
    w, h = page_size
    boxes = []
    for _ in range(n):
        if degradation == "watermark":
            boxes.append([w * 0.1, h * 0.25, w * 0.9, h * 0.75])
        elif degradation == "stain":
            r = rng.uniform(40.0, 110.0)
            cx, cy = rng.uniform(MARGIN, w - MARGIN), rng.uniform(MARGIN, h - MARGIN)
            boxes.append([cx - r, cy - r * 0.7, cx + r, cy + r * 0.7])
        elif degradation == "stamp":
            r = rng.uniform(35.0, 60.0)
            cx, cy = rng.uniform(MARGIN + r, w - MARGIN - r), rng.uniform(MARGIN + r, h - MARGIN - r)
            boxes.append([cx - r, cy - r, cx + r, cy + r])
    return boxes


def render_document(doc: dict, controls: RenderControls, out_dir, page_size=(612.0, 792.0),
                    seed: int = 0, scale: float = 2.0) -> List[RenderedPage]:
    """Lay out paragraphs and HTML tables page by page, rasterise them with the
    requested border style, paint the degradation on top, and record the text
    layer (word boxes) and the artefact boxes."""
    from pathlib import Path

    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    font = _font(FONT_SIZE, scale)
    width = page_size[0] - 2 * MARGIN
    pages: List[RenderedPage] = []
    for p in doc["pages"]:
        blocks, words, y = [], [], MARGIN
        tables = list(p.get("tables", []))
        for i, para in enumerate(p.get("paragraphs", [])):
            blk, ws = _layout_paragraph(para, MARGIN, y, width, font, scale, len(blocks))
            blocks.append(blk)
            words += ws
            y = blk["bbox"][3] + 10.0
            if i < len(tables):
                t = tables[i]
                blk, ws = _layout_table(t["table_html"], f"{p['page_index']}_{i}", MARGIN, y, width, font, scale, len(blocks))
                blocks.append(blk)
                words += ws
                y = blk["bbox"][3] + 12.0
        for i in range(len(p.get("paragraphs", [])), len(tables)):        # tables without a leading paragraph
            blk, ws = _layout_table(tables[i]["table_html"], f"{p['page_index']}_{i}", MARGIN, y, width, font, scale, len(blocks))
            blocks.append(blk)
            words += ws
            y = blk["bbox"][3] + 12.0
        artifacts = [] if controls.degradation == "none" else \
            _page_artifacts(rng, controls.degradation, page_size, n=1 if controls.degradation == "watermark" else rng.randint(1, 3))
        image_path = out_dir / f"page_{p['page_index']}.png"
        draw_page(blocks, artifacts, controls, page_size, image_path, font=font, scale=scale, rng=rng)
        pages.append(RenderedPage(p["page_index"], str(image_path), page_size, blocks, words, artifacts))
    return pages


def draw_page(blocks: List[dict], artifacts: List[List[float]], controls: RenderControls, page_size, out_path,
              font=None, scale: float = 2.0, rng: Optional[random.Random] = None) -> None:
    """Rasterise one page with PIL: wrapped text, ruled tables in the requested
    border style, and the degradation overlay (watermark / stain / stamp)."""
    from PIL import Image, ImageDraw

    rng = rng or random.Random(0)
    font = font or _font(FONT_SIZE, scale)
    S = lambda v: v * scale  # noqa: E731
    img = Image.new("RGB", (int(S(page_size[0])), int(S(page_size[1]))), "white")
    draw = ImageDraw.Draw(img, "RGBA")

    for b in blocks:
        if b["type"] == "paragraph":
            for line_no, line in enumerate(_wrap(font, b["text"], b["bbox"][2] - b["bbox"][0], scale)):
                draw.text((S(b["bbox"][0]), S(b["bbox"][1] + line_no * LINE_H)), line, fill="black", font=font)
            continue
        # cell text
        for c in b.get("cells", []):
            for line_no, line in enumerate(c["lines"]):
                draw.text((S(c["bbox"][0] + CELL_PAD), S(c["bbox"][1] + CELL_PAD + line_no * LINE_H)), line,
                          fill="black", font=font)
        # rulings
        x0, y0, x1, y1 = b["bbox"]
        row_y = b.get("row_y", [y0, y1])
        header_end = row_y[min(b.get("header_rows", 0), len(row_y) - 1)] if b.get("header_rows") else None
        border = controls.border
        if border == "full":
            draw.rectangle([S(x0), S(y0), S(x1), S(y1)], outline="black", width=1)
            for c in b.get("cells", []):                                   # spanned cells keep one box
                draw.rectangle([S(v) for v in c["bbox"]], outline="black", width=1)
        elif border == "three-line":
            for yy in (y0, y1) + ((header_end,) if header_end else ()):
                draw.line([S(x0), S(yy), S(x1), S(yy)], fill="black", width=2 if yy in (y0, y1) else 1)
        elif border == "header-only":
            if header_end:
                draw.line([S(x0), S(header_end), S(x1), S(header_end)], fill="black", width=1)
        else:  # "other": light horizontal rulings only, no outer frame
            for yy in row_y[1:-1]:
                draw.line([S(x0), S(yy), S(x1), S(yy)], fill=(180, 180, 180), width=1)

    for a in artifacts:
        ax0, ay0, ax1, ay1 = [S(v) for v in a]
        if controls.degradation == "watermark":
            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ld = ImageDraw.Draw(layer)
            big = _font(FONT_SIZE * 6, scale)
            ld.text(((ax0 + ax1) / 2 - S(150), (ay0 + ay1) / 2 - S(30)), "CONFIDENTIAL", fill=(120, 120, 200, 90), font=big)
            layer = layer.rotate(30, center=((ax0 + ax1) / 2, (ay0 + ay1) / 2))
            img.paste(layer, (0, 0), layer)
        elif controls.degradation == "stain":
            for k in range(3):                                            # overlapping blobs
                jx, jy = rng.uniform(-0.15, 0.15) * (ax1 - ax0), rng.uniform(-0.15, 0.15) * (ay1 - ay0)
                draw.ellipse([ax0 + jx, ay0 + jy, ax1 + jx, ay1 + jy], fill=(150, 100, 40, 90 + 25 * k))
        elif controls.degradation == "stamp":
            draw.ellipse([ax0, ay0, ax1, ay1], outline=(200, 40, 40, 190), width=int(3 * scale))
            draw.ellipse([ax0 + S(6), ay0 + S(6), ax1 - S(6), ay1 - S(6)], outline=(200, 40, 40, 190), width=int(1 * scale))
            draw.text(((ax0 + ax1) / 2 - S(22), (ay0 + ay1) / 2 - S(6)), "APPROVED", fill=(200, 40, 40, 190), font=font)
    img.save(out_path)


# --------------------------------------------------------------- outputs
def crop_tables(pages: List[RenderedPage], doc_dir, scale: float = 2.0) -> List[dict]:
    """Cut every table box out of its rendered page (``tables/page_<P>_t<ID>.png``)
    and write the source HTML next to it (``table_html/``)."""
    from pathlib import Path
    from PIL import Image

    doc_dir = Path(doc_dir)
    (doc_dir / "tables").mkdir(parents=True, exist_ok=True)
    (doc_dir / "table_html").mkdir(parents=True, exist_ok=True)
    records = []
    for p in pages:
        page_img = Image.open(p.image_path)
        for b in p.blocks:
            if b["type"] != "table":
                continue
            stem = f"page_{p.page_index}_t{b['table_id']}"
            x0, y0, x1, y1 = b["bbox"]
            pad = 4.0
            page_img.crop((int((x0 - pad) * scale), int((y0 - pad) * scale),
                           int((x1 + pad) * scale), int((y1 + pad) * scale))).save(doc_dir / "tables" / f"{stem}.png")
            (doc_dir / "table_html" / f"{stem}.html").write_text(b["html"], encoding="utf-8")
            records.append({"table_id": b["table_id"], "page_num": p.page_index, "bbox": list(b["bbox"]),
                            "table_png_relpath": f"tables/{stem}.png"})
    return records


def source_records(pages: List[RenderedPage]) -> Dict[str, list]:
    """What is retained per document: ordered page images, table boxes and
    source HTML, the artefact boxes, and the text layer under them."""
    return {"pages": [p.image_path for p in pages],
            "tables": [{"page": p.page_index, "table_id": b["table_id"], "html": b["html"], "bbox": b["bbox"]}
                       for p in pages for b in p.blocks if b["type"] == "table"],
            "artifacts": [{"page": p.page_index, "boxes": p.artifacts} for p in pages]}


def table_ocr_text(html: str) -> str:
    """Characters-only text of a table (no structure), one row per line."""
    cells, n_rows, _ = html_to_grid(html)
    rows: Dict[int, List[str]] = {}
    for c in sorted(cells, key=lambda c: (c.row, c.col)):
        rows.setdefault(c.row, []).append(c.text)
    return "\n".join(" ".join(rows.get(r, [])) for r in range(n_rows))
