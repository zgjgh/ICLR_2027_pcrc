"""Front-end document parsing (Section 3 and Section 5.1, "Upstream parsing").

Layout detection and OCR are replaceable front-end components:

* PubConTab pages are parsed with PP-Structure from the PaddleOCR toolkit
  (layout regions + OCR).
* FinConTab and DegConTab source PDFs carry a machine-readable text layer,
  so text and layout are read with PyMuPDF instead of running optical
  recognition. For DegConTab, text under the rendered watermarks, stains,
  and stamps is masked out, so degraded regions are observed only through
  the image (``mask_degraded_text``).

The output of this stage is the ``segment_json`` block format consumed by
``pcrc.data.corpus``: one JSON per page with ``{key: {type, bbox, ocr}}``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

TEXT_TYPES = {"paragraph", "caption", "title", "text", "list", "footnote"}


# ---------------------------------------------------------------- helpers
def _rect_intersection(a: Sequence[float], b: Sequence[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _rect_area(a: Sequence[float]) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def mask_degraded_text(words: List[dict], artifacts: Iterable[Sequence[float]],
                       min_overlap: float = 0.3) -> List[dict]:
    """Drop text-layer words that lie under an injected visual artifact.

    ``words`` are PyMuPDF-style word records ``{"bbox": [x0,y0,x1,y1], "text": str}``;
    ``artifacts`` are the bounding boxes of the watermarks, stains, and stamps
    that the DegConTab renderer applied to the page. A word is removed when at
    least ``min_overlap`` of its area is covered by an artifact, so that the
    text layer cannot reveal characters that are unreadable in the image.
    """
    kept = []
    arts = [list(a) for a in artifacts]
    for w in words:
        area = _rect_area(w["bbox"]) or 1e-9
        covered = max((_rect_intersection(w["bbox"], a) / area for a in arts), default=0.0)
        if covered < min_overlap:
            kept.append(w)
    return kept


def words_to_text(words: List[dict]) -> str:
    """Join word records in reading order into one unstructured line sequence."""
    rows: Dict[int, List[dict]] = {}
    for w in words:
        rows.setdefault(int(round(w["bbox"][1] / 4.0)), []).append(w)
    lines = []
    for y in sorted(rows):
        lines.append(" ".join(w["text"] for w in sorted(rows[y], key=lambda r: r["bbox"][0])))
    return "\n".join(lines)


# ------------------------------------------------------------ PyMuPDF path
def parse_pdf_page(pdf_path: str, page_index: int, layout_blocks: List[dict],
                   artifacts: Optional[List[Sequence[float]]] = None) -> Dict[str, dict]:
    """Read the text layer of one page and attach it to detected layout blocks.

    ``layout_blocks``: ``[{"type": "paragraph"|"table"|..., "bbox": [...], "table_id": ...}]``
    from the layout detector. ``artifacts`` (DegConTab only) are masked first.
    Returns ``{local_key: {type, bbox, ocr, table_id?}}``.
    """
    import fitz  # PyMuPDF

    page = fitz.open(pdf_path)[page_index]
    words = [{"bbox": [w[0], w[1], w[2], w[3]], "text": w[4]} for w in page.get_text("words")]
    if artifacts:
        words = mask_degraded_text(words, artifacts)
    blocks: Dict[str, dict] = {}
    counters: Dict[str, int] = {}
    for blk in layout_blocks:
        inside = [w for w in words if _rect_intersection(w["bbox"], blk["bbox"]) > 0.5 * (_rect_area(w["bbox"]) or 1e-9)]
        t = blk["type"]
        counters[t] = counters.get(t, 0) + 1
        key = f"{t}_{counters[t] - 1}"
        rec = {"type": t, "bbox": list(blk["bbox"]), "ocr": words_to_text(inside)}
        if t == "table":
            rec["table_id"] = blk.get("table_id")
        blocks[key] = rec
    return blocks


# ----------------------------------------------------------- PP-Structure
def parse_image_page(image_path: str) -> Dict[str, dict]:
    """Run PP-Structure (PaddleOCR) on a rendered page: layout regions + OCR."""
    from paddleocr import PPStructure  # type: ignore

    engine = PPStructure(show_log=False, table=False, ocr=True)
    blocks: Dict[str, dict] = {}
    counters: Dict[str, int] = {}
    for region in engine(image_path):
        t = "table" if region["type"] == "table" else ("paragraph" if region["type"] in ("text", "list") else region["type"])
        counters[t] = counters.get(t, 0) + 1
        text = "\n".join(line["text"] for line in region.get("res", []) if isinstance(line, dict) and "text" in line)
        blocks[f"{t}_{counters[t] - 1}"] = {"type": t, "bbox": list(region["bbox"]), "ocr": text}
    return blocks


def write_segment_json(out_dir: Path, doc_id: str, page: int, blocks: Dict[str, dict],
                       page_size: Sequence[float]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{doc_id}_p{page}.json"
    payload = {"meta": {"doc_id": doc_id, "page": page, "page_size": list(page_size)},
               "blocks": {f"{doc_id}_p{page}__{k}": v for k, v in blocks.items()}}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return p
