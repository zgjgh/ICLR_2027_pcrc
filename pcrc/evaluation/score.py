"""Aggregate inference records into the paper's tables.

Records are JSONL files written by the experiment scripts, one line per
(dataset, method, document, condition, table) in ``TableOutcome`` format,
under ``<PCRC_OUTPUT_ROOT>/records/<dataset>/<method>.jsonl``.

``score_dataset`` produces, per method: judgment P/R/F1 (Table 7),
composite P_c/R_c/F1 (Table 2), cost (Table 3 / Table 9 / Table 10 /
Table 12 / Table 13), and the per-category breakdown (Table 8).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .. import config
from ..data.corpus import load_document
from .metrics import CostScores, QualityScores, cost, judgment_and_composite, per_category


def records_dir(dataset: str) -> Path:
    return config.OUTPUT_ROOT / "records" / config.DATASET_ALIASES[dataset.lower()]


def write_records(dataset: str, method: str, outcomes: Iterable, append: bool = False) -> Path:
    p = records_dir(dataset) / f"{method}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a" if append else "w", encoding="utf-8") as f:
        for o in outcomes:
            f.write(json.dumps(o.to_json() if hasattr(o, "to_json") else o, ensure_ascii=False) + "\n")
    return p


def read_records(dataset: str, method: str) -> List[dict]:
    p = records_dir(dataset) / f"{method}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def reference_html(dataset: str, doc_ids: Iterable[str]) -> Dict[tuple, str]:
    ref: Dict[tuple, str] = {}
    for d in set(doc_ids):
        doc = load_document(dataset, d)
        if doc:
            for t in doc.tables:
                ref[(d, str(t.table_id))] = t.html_ref
    return ref


def score_dataset(dataset: str, methods: Optional[List[str]] = None, subset_ids: Optional[set] = None) -> Dict[str, dict]:
    """Per-method scores for one dataset. ``subset_ids`` restricts to condition ids."""
    if methods is None:
        methods = sorted(p.stem for p in records_dir(dataset).glob("*.jsonl"))
    out: Dict[str, dict] = {}
    for m in methods:
        recs = read_records(dataset, m)
        if subset_ids is not None:
            recs = [r for r in recs if r["condition_id"] in subset_ids]
        if not recs:
            continue
        ref = reference_html(dataset, (r["doc_id"] for r in recs))
        q: QualityScores = judgment_and_composite(recs, ref)
        c: CostScores = cost(recs)
        cats = {k: v.as_dict() for k, v in per_category(recs, ref).items()}
        out[m] = {"quality": q.as_dict(), "cost": c.as_dict(), "per_category": cats, "n_records": len(recs)}
    return out


def format_row(name: str, s: dict) -> str:
    q, c = s["quality"], s["cost"]
    return (f"{name:34s} P {q['precision']:.3f} R {q['recall']:.3f} F1 {q['f1']:.3f} | "
            f"Pc {q['precision_c']:.3f} Rc {q['recall_c']:.3f} F1c {q['f1_c']:.3f} | "
            f"Pre {c['prefill_k']:.2f}k Dec {c['decode_k']:.2f}k Tot {c['total_k']:.2f}k R.R. {c['recomputation_ratio']:.2f}%")


def save_scores(dataset: str, scores: Dict[str, dict], tag: str = "scores") -> Path:
    p = config.OUTPUT_ROOT / "scores" / f"{config.DATASET_ALIASES[dataset.lower()]}_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    return p
