"""Non-VLM HTML extractors used by the modular pipelines (Section 5.1).

* TATR   the Table Transformer structure-recognition model, run on the cropped
         table image; cells are assembled from the detected rows, columns, and
         spanning cells, and filled with the word boxes of the OCR/text layer
         (recognised on the crop with PaddleOCR when none are supplied).
* MinerU the non-VLM image-based document/table parsing pipeline
         (``pipeline`` backend), not the VL variant.

Both are used off the shelf with their released weights and default
configurations; they only produce HTML and never judge conditions.
"""
from __future__ import annotations

from typing import List, Optional

TATR_HF_ID = "microsoft/table-transformer-structure-recognition"


class TATRExtractor:
    def __init__(self, weights: Optional[str] = None, device: str = "cuda:0", threshold: float = 0.5):
        """``weights``: HF id or local directory of the structure-recognition
        checkpoint (default: the released ``pubtables1m`` DETR-R18 model)."""
        from transformers import AutoModelForObjectDetection, DetrImageProcessor

        src = weights or TATR_HF_ID
        self.model = AutoModelForObjectDetection.from_pretrained(src).to(device).eval()
        self.processor = DetrImageProcessor.from_pretrained(src)
        self.device = device
        self.threshold = threshold
        self._ocr = None

    # ------------------------------------------------------------- OCR
    def word_boxes(self, table_image) -> List[dict]:
        """Recognise words on the crop when the caller has no text layer for it."""
        import numpy as np
        if self._ocr is None:
            from paddleocr import PaddleOCR  # type: ignore
            self._ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        words = []
        for line in (self._ocr.ocr(np.asarray(table_image), cls=False) or [[]])[0] or []:
            quad, (text, _) = line
            xs, ys = [p[0] for p in quad], [p[1] for p in quad]
            words.append({"bbox": [min(xs), min(ys), max(xs), max(ys)], "text": text})
        return words

    # ------------------------------------------------------- structure
    def detect(self, table_image) -> dict:
        """Rows, columns, and spanning cells (image coordinates) above ``threshold``."""
        import torch

        inputs = self.processor(images=table_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        target = torch.tensor([table_image.size[::-1]], device=self.device)
        objs = self.processor.post_process_object_detection(out, threshold=self.threshold, target_sizes=target)[0]
        labels = self.model.config.id2label
        by_label = {}
        for box, lab in zip(objs["boxes"].tolist(), objs["labels"].tolist()):
            by_label.setdefault(labels[lab], []).append(box)
        return {
            "rows": sorted(by_label.get("table row", []), key=lambda b: b[1]),
            "columns": sorted(by_label.get("table column", []), key=lambda b: b[0]),
            "spans": by_label.get("table spanning cell", []) + by_label.get("table projected row header", []),
        }

    def extract(self, table_image, word_boxes: Optional[List[dict]] = None) -> str:
        """Return HTML for one cropped table."""
        struct = self.detect(table_image)
        words = word_boxes if word_boxes is not None else self.word_boxes(table_image)
        return cells_to_html(struct["rows"], struct["columns"], words, struct["spans"])


def _centre_in(b, cell) -> bool:
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return cell[0] <= cx <= cell[2] and cell[1] <= cy <= cell[3]


def cells_to_html(rows: List[List[float]], cols: List[List[float]], words: List[dict],
                  spans: Optional[List[List[float]]] = None) -> str:
    """Assign words to the row x column grid, merge grid cells covered by a
    spanning-cell box into one ``rowspan``/``colspan`` cell, and emit HTML."""
    n_r, n_c = len(rows), len(cols)
    if n_r == 0 or n_c == 0:
        return "<table></table>"
    grid = [[[c[0], r[1], c[2], r[3]] for c in cols] for r in rows]
    owner = [[(i, j) for j in range(n_c)] for i in range(n_r)]        # top-left cell of the merged block
    extent = {}                                                        # (i, j) -> (rowspan, colspan)
    for sp in spans or []:
        covered = [(i, j) for i in range(n_r) for j in range(n_c) if _centre_in(grid[i][j], sp)]
        if len(covered) < 2:
            continue
        i0, j0 = min(i for i, _ in covered), min(j for _, j in covered)
        i1, j1 = max(i for i, _ in covered), max(j for _, j in covered)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                owner[i][j] = (i0, j0)
        extent[(i0, j0)] = (i1 - i0 + 1, j1 - j0 + 1)
    ordered = sorted(words, key=lambda w: (round(w["bbox"][1]), w["bbox"][0]))
    html = ["<table>"]
    for i in range(n_r):
        html.append("<tr>")
        for j in range(n_c):
            if owner[i][j] != (i, j):
                continue
            rs, cs = extent.get((i, j), (1, 1))
            box = [grid[i][j][0], grid[i][j][1], grid[i][j + cs - 1][2], grid[i + rs - 1][j][3]]
            text = " ".join(w["text"] for w in ordered if _centre_in(w["bbox"], box))
            attrs = (f' rowspan="{rs}"' if rs > 1 else "") + (f' colspan="{cs}"' if cs > 1 else "")
            html.append(f"<td{attrs}>{text}</td>")
        html.append("</tr>")
    html.append("</table>")
    return "".join(html)


class MinerUExtractor:
    """Reads the HTML MinerU wrote for each table crop in a batch run
    (``mineru -p <crops> -o <out> -b pipeline -m ocr``)."""

    def __init__(self, mineru_out_dir):
        from pathlib import Path
        self.out_dir = Path(mineru_out_dir)

    def extract(self, image_path: str) -> str:
        from pathlib import Path
        import json
        stem = Path(image_path).stem
        for cand in self.out_dir.rglob(f"{stem}*_content_list.json"):
            for block in json.loads(cand.read_text(encoding="utf-8")):
                if block.get("type") == "table" and block.get("table_body"):
                    return block["table_body"]
        return ""


def make_extractor(name: str, **kwargs):
    return TATRExtractor(**kwargs) if name == "tatr" else MinerUExtractor(**kwargs)
