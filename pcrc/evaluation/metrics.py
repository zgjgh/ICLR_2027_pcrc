"""Evaluation criteria (Section 3.3) and cost accounting (Section 5.3).

Judgment quality   micro precision / recall / F1 on the positive class over
                   all (condition, candidate-table) pairs.
Extraction quality EQ = cell-count weighted TEDS on true-positive tables.
Composite quality  P_c = P * EQ, R_c = R * EQ, and their harmonic mean F1
                   (Table 2, "F1-TEDS" in the appendix tables).
Cost               per-condition prefill / decode / total tokens (thousands)
                   averaged over conditions, and the recomputation ratio
                   R.R. = sum(prefill_actual) / sum(prefill_nocache) (%).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .teds import cell_count, teds_score


@dataclass
class QualityScores:
    tp: int
    fp: int
    fn: int
    tn: int
    eq: float                  # cell-count weighted TEDS on TP
    precision: float
    recall: float
    f1: float
    precision_c: float
    recall_c: float
    f1_c: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def judgment_and_composite(records: Iterable[dict], ref_html: Dict[tuple, str]) -> QualityScores:
    """``records``: TableOutcome dicts; ``ref_html``: (doc_id, table_id) -> reference HTML."""
    tp = fp = fn = tn = 0
    w_sum = teds_w = 0.0
    for r in records:
        g, p = r["gold"] == "yes", r["predicted"] == "yes"
        if g and p:
            tp += 1
            ref = ref_html.get((r["doc_id"], str(r["table_id"])), "")
            w = max(cell_count(ref), 1)
            w_sum += w
            teds_w += w * teds_score(r.get("html_pred"), ref)
        elif p:
            fp += 1
        elif g:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    eq = teds_w / w_sum if w_sum else 0.0
    pc, rc = precision * eq, recall * eq
    return QualityScores(tp, fp, fn, tn, eq, precision, recall, _f1(precision, recall), pc, rc, _f1(pc, rc))


@dataclass
class CostScores:
    prefill_k: float          # per-condition prefill tokens actually encoded, thousands
    decode_k: float
    total_k: float
    recomputation_ratio: float
    n_conditions: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def cost(records: Iterable[dict]) -> CostScores:
    per_cond: Dict[tuple, List[int]] = defaultdict(lambda: [0, 0, 0])
    for r in records:
        k = (r["doc_id"], r["condition_id"])
        per_cond[k][0] += int(r.get("prefill_actual", 0))
        per_cond[k][1] += int(r.get("prefill_nocache", 0))
        per_cond[k][2] += int(r.get("decode", 0))
    n = len(per_cond) or 1
    pre = sum(v[0] for v in per_cond.values()) / n
    noc = sum(v[1] for v in per_cond.values())
    dec = sum(v[2] for v in per_cond.values()) / n
    rr = 100.0 * sum(v[0] for v in per_cond.values()) / noc if noc else float("nan")
    return CostScores(pre / 1000, dec / 1000, (pre + dec) / 1000, rr, len(per_cond))


def per_category(records: Iterable[dict], ref_html: Dict[tuple, str], categories=("C1", "C2", "C3", "C4", "C5")) -> Dict[str, QualityScores]:
    """Judgment / composite quality computed within each condition stratum (Table 8)."""
    by_cat: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_cat[r.get("category", "")].append(r)
    return {c: judgment_and_composite(by_cat.get(c, []), ref_html) for c in categories}


def average_over_datasets(per_dataset: List[Dict[str, QualityScores]]) -> Dict[str, dict]:
    """Table 8: per-stratum scores averaged (unweighted) over the three datasets."""
    out: Dict[str, dict] = {}
    for c in per_dataset[0]:
        out[c] = {"f1": sum(d[c].f1 for d in per_dataset) / len(per_dataset),
                  "f1_c": sum(d[c].f1_c for d in per_dataset) / len(per_dataset)}
    return out
