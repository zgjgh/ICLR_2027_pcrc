"""Relevance index construction (Section 3.1, "Relevance Matrix", Eq. 2-3).

Given the relevance matrix R = [r(P_j, T_i)], PCRC preprocessing produces

    I(D) = ( S(D), {E(T_i)}_{i=1..n} ).

* Shared related paragraphs. For each paragraph the relevance is averaged over
  all tables, g(P_j) = (1/n) sum_i r(P_j, T_i). Paragraphs with g(P_j) >=
  tau_shared enter S(D), truncated to the K_shared highest g.
* Exclusive related paragraphs. For each table,
  E(T_i) = TopK_{K_excl} { P_j not in S(D) : r(P_j, T_i) >= tau_excl },
  ordered by descending r so that progressive rounds append the most
  relevant paragraph first.

The variants of Appendix A.3.4 are obtained by ``mode``: ``random`` draws
K_shared + K_excl paragraphs at random (no ranking), ``all`` puts every
paragraph of the document into S(D) and leaves E(T_i) empty.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np

from .. import config
from ..data.schema import Document, RelevanceIndex
from .relevance import relevance_matrix, spatial_proximity
from .embedding import semantic_similarity


def _reading_order(doc: Document, keys: List[str]) -> List[str]:
    pos = {p.key: (p.page, p.bbox[1]) for p in doc.paragraphs}
    return sorted(keys, key=lambda k: pos.get(k, (0, 0)))


def build_index(doc: Document, par_emb: Optional[np.ndarray], tab_emb: Optional[np.ndarray],
                cfg: config.SelectionConfig = config.DEFAULT_SELECTION, mode: str = "pcrc",
                seed: int = 0) -> RelevanceIndex:
    """Compute I(D) for one document.

    ``mode``: ``pcrc`` (default), ``random``, ``all``. The relevance factor
    combination (product / semantic_only / spatial_only) comes from ``cfg.relevance``.
    """
    keys = [p.key for p in doc.paragraphs]
    n_par, n_tab = len(doc.paragraphs), len(doc.tables)

    if mode == "all":
        return RelevanceIndex(doc.doc_id, _reading_order(doc, keys), {t.key: [] for t in doc.tables})

    if mode == "random":
        rng = random.Random(f"{seed}:{doc.doc_id}")
        budget = cfg.k_shared + cfg.k_excl
        picked = rng.sample(keys, min(budget, len(keys)))
        rng.shuffle(picked)   # random order, delivered in a single round (Table 9)
        return RelevanceIndex(doc.doc_id, picked, {t.key: [] for t in doc.tables})

    if n_par == 0 or n_tab == 0:
        return RelevanceIndex(doc.doc_id, [], {t.key: [] for t in doc.tables})

    s_sem = semantic_similarity(par_emb, tab_emb)             # (n_par, n_tab)
    s_spa = spatial_proximity(doc, lam=cfg.lam)               # (n_par, n_tab)
    R = relevance_matrix(s_sem, s_spa, cfg.relevance)

    # --- shared: mean over tables, threshold, budget (Eq. 2)
    g = R.mean(axis=1)
    shared_idx = [j for j in np.argsort(-g) if g[j] >= cfg.tau_shared][: cfg.k_shared]
    shared = set(shared_idx)
    shared_keys = _reading_order(doc, [keys[j] for j in shared_idx])

    # --- exclusive: per table, exclude S(D), threshold, top-K by r (Eq. 3)
    exclusive: Dict[str, List[str]] = {}
    scores: Dict[str, Dict[str, float]] = {}
    for i, t in enumerate(doc.tables):
        cand = [j for j in np.argsort(-R[:, i]) if j not in shared and R[j, i] >= cfg.tau_excl]
        exclusive[t.key] = [keys[j] for j in cand[: cfg.k_excl]]
        scores[t.key] = {keys[j]: float(R[j, i]) for j in range(n_par)}

    return RelevanceIndex(doc.doc_id, shared_keys, exclusive, scores,
                          doc_scores={keys[j]: float(g[j]) for j in range(n_par)})


def restrict_index(index: RelevanceIndex, variant: str) -> RelevanceIndex:
    """Evidence-type ablations of Table 9: ``shared_only`` / ``exclusive_only``."""
    if variant == "shared_only":
        return RelevanceIndex(index.doc_id, index.shared_keys, {k: [] for k in index.exclusive_keys},
                              index.scores, index.doc_scores)
    if variant == "exclusive_only":
        return RelevanceIndex(index.doc_id, [], index.exclusive_keys, index.scores, index.doc_scores)
    return index
