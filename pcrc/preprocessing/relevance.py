"""Paragraph-table relevance score (Section 3.1, Eq. 1) and its two factors.

    r(P_j, T_i) = s_sem(P_j, T_i) * s_spa(P_j, T_i)

* s_sem = max{0, cos(phi(OCR(P_j)), phi(OCR(T_i)))}          (embedding.py)
* s_spa = exp(-lam * d_hat_ij),  d_hat_ij = d_ij / d_max in [0, 1]
  where d_ij is the distance between the bounding-box centres of P_j and
  T_i, pages being stacked vertically so that distances are defined across
  pages, and d_max is the largest centre-to-centre distance among all
  paragraph-table pairs of the document (Appendix A.3.6). lam = 1 is the
  default; Table 12 sweeps lam.

Both factors lie in [0, 1]. A weak semantic match drives the score toward
zero, while the normalised spatial factor scales it within [1/e, 1].
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ..data.schema import Document


def _centre(bbox: Sequence[float]) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float64)


def _page_offset(doc: Document, page: int) -> float:
    """Vertical offset of a page when pages are stacked in reading order."""
    off = 0.0
    for p in range(1, page):
        off += doc.page_sizes.get(p, [612.0, 792.0])[1]
    return off


def centre_distances(doc: Document) -> np.ndarray:
    """d[j, i]: Euclidean distance between paragraph j and table i centres."""
    P = np.array([_centre(p.bbox) + [0.0, _page_offset(doc, p.page)] for p in doc.paragraphs]) \
        if doc.paragraphs else np.zeros((0, 2))
    T = np.array([_centre(t.bbox) + [0.0, _page_offset(doc, t.page)] for t in doc.tables]) \
        if doc.tables else np.zeros((0, 2))
    if len(P) == 0 or len(T) == 0:
        return np.zeros((len(P), len(T)))
    return np.linalg.norm(P[:, None, :] - T[None, :, :], axis=-1)


def spatial_proximity(doc: Document, lam: float = 1.0) -> np.ndarray:
    """s_spa[j, i] = exp(-lam * d_hat_ij) with per-document normalisation."""
    d = centre_distances(doc)
    d_max = float(d.max()) if d.size else 1.0
    d_hat = d / d_max if d_max > 0 else d
    return np.exp(-lam * d_hat)


def relevance_matrix(s_sem: np.ndarray, s_spa: np.ndarray, mode: str = "product") -> np.ndarray:
    """R[j, i] = r(P_j, T_i). ``mode`` selects the Table 9 variants."""
    if mode == "product":
        return s_sem * s_spa
    if mode == "semantic_only":
        return s_sem
    if mode == "spatial_only":
        return s_spa
    raise ValueError(f"unknown relevance mode: {mode}")


def factor_quantiles(values: np.ndarray, qs=(5, 25, 50, 75, 95)) -> Dict[str, float]:
    """Empirical distribution summary used by Table 11 / Table 12 (spread = q95/q05)."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    out = {f"q{q:02d}": float(np.percentile(flat, q)) for q in qs} if flat.size else {f"q{q:02d}": float("nan") for q in qs}
    out["spread"] = out["q95"] / out["q05"] if out.get("q05", 0) > 0 else float("nan")
    return out
