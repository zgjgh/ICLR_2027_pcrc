"""TEDS (tree-edit-distance similarity) over HTML tables, via the official
PubTabNet implementation bundled under ``deps/teds``.

Extraction quality is scored only on true positives (Section 3.3): false
positives are already penalised by precision and false negatives by recall.
EQ is the cell-count weighted TEDS over TP tables, cell count being the
number of ``<td>`` cells of the reference table.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

_DEPS = Path(__file__).resolve().parents[2] / "deps"
if str(_DEPS) not in sys.path:
    sys.path.insert(0, str(_DEPS))

_TEDS = None


def _teds():
    global _TEDS
    if _TEDS is None:
        from teds.teds import TEDS  # deps/teds/teds.py
        _TEDS = TEDS(structure_only=False, n_jobs=1)
    return _TEDS


def normalise_html(html: str) -> str:
    """Compact form accepted by TEDS: one <table>, <td> cells, no wrappers."""
    h = re.sub(r"```[a-z]*", "", html or "").strip()
    h = re.sub(r"<(/?)th\b", r"<\1td", h)
    h = re.sub(r"</?(thead|tbody|html|body)[^>]*>", "", h)
    start, end = h.lower().find("<table"), h.lower().rfind("</table>")
    if start >= 0 and end > start:
        h = h[start:end + len("</table>")]
    return h if h.startswith("<table") else "<table></table>"


def cell_count(html: str) -> int:
    return len(re.findall(r"<td\b", html or ""))


def teds_score(pred_html: Optional[str], ref_html: str) -> float:
    if not pred_html:
        return 0.0
    try:
        return float(_teds().evaluate(f"<html><body>{normalise_html(pred_html)}</body></html>",
                                      f"<html><body>{normalise_html(ref_html)}</body></html>"))
    except Exception:
        return 0.0
