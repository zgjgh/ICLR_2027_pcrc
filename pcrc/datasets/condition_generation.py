"""Source-grounded condition construction (Section 4, Appendix A.1.5-A.1.7).

* Five balanced strata C1-C5 (Table 6), 20% each; every generation call
  targets one stratum and calls are scheduled so that the pool follows the
  target distribution.
* The generator (GPT-5.2) sees only the reference HTML of all tables of a
  document, in page order, never the paragraphs or images. It returns one
  condition, its category, and a yes/no label for every table, so a
  condition may match zero, one, or multiple tables.
* The LLM output is a proposal, not ground truth: automatic checks remove
  malformed outputs, duplicates, ambiguous or under-specified wording, and
  label inconsistencies; a stratified sample is then validated by human
  annotators against the HTML and the rendered document.
* Splits are made at the document level.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import config

GENERATOR_MODEL = "gpt-5.2"

CONDITION_SYSTEM = "You are constructing data for conditioned table extraction."

CONDITION_PROMPT = """[Input]
The document contains multiple tables. For each table, you are given only its reference HTML. The tables are listed in document page order.
The target condition type for this call is: {TARGET_CONDITION_TYPE}.

[Goal]
Generate one natural-language condition of the target type for conditioned table extraction in this document. Assign a yes/no label to every table. The condition may match zero, one, or multiple tables.

[Instructions]
1. Generate exactly one condition for the document.
2. The condition must follow the supplied target condition type.
3. Each condition must be answerable from the provided table HTML strings.
4. Do not rely on external knowledge.
5. Do not mention table IDs, page numbers, file names, or annotation fields.
6. Avoid trivial conditions that match all tables unless the evidence makes such a condition meaningful.
7. Avoid conditions that are ambiguous, underspecified, or impossible to judge from the provided tables.
8. For each condition, label every table as "yes" or "no".
9. Return only valid JSON.

[Output Schema]
{{
  "condition": "...",
  "category": "{TARGET_CONDITION_TYPE}",
  "table_labels": [
    {{"table_id": "T1", "label": "yes|no", "evidence": "..."}},
    {{"table_id": "T2", "label": "yes|no", "evidence": "..."}}
  ]
}}

[Document Tables]
{DOCUMENT_TABLES}"""

STRATA = {
    "C1": "Lookup & Filter: literal lookup or filtering over table entries, headers, or explicit attributes.",
    "C2": "Numerical Reasoning: aggregation, comparison, extrema, ordinal selection, or trend checking, where the central evidence is quantitative.",
    "C3": "Multi-Constraint Conjunction: several independent constraints must hold simultaneously, without an intermediate derived result.",
    "C4": "Multi-Step Reasoning: operator chains in which an intermediate result is needed before the final judgment.",
    "C5": "Structural Conditions: conditions on table structure, such as hierarchy, row/column semantics, aggregation pattern, or organization form.",
}


def schedule_strata(n_calls: int, seed: int = 0) -> List[str]:
    """Stratum per call so that the pool follows the 20% target shares."""
    keys = list(config.CONDITION_CATEGORIES)
    plan = [keys[i % len(keys)] for i in range(n_calls)]
    random.Random(seed).shuffle(plan)
    return plan


def document_tables_block(tables: List[dict]) -> str:
    """``tables``: [{"table_id": ..., "html": ...}] in page order -> T1..Tn listing."""
    return "\n\n".join(f"T{i + 1}:\n{t['html'].strip()}" for i, t in enumerate(tables))


def build_prompt(tables: List[dict], stratum: str) -> List[dict]:
    user = CONDITION_PROMPT.format(TARGET_CONDITION_TYPE=f"{stratum} {STRATA[stratum]}",
                                   DOCUMENT_TABLES=document_tables_block(tables))
    return [{"role": "system", "content": CONDITION_SYSTEM}, {"role": "user", "content": user}]


# ------------------------------------------------------------ validation
@dataclass
class CandidateCondition:
    doc_id: str
    stratum: str
    condition: str
    labels: Dict[str, str]                 # table_id -> yes|no
    evidence: Dict[str, str] = field(default_factory=dict)
    rejected: Optional[str] = None


_VAGUE = re.compile(r"\b(some|several|many|few|various|etc\.?|and so on|maybe|possibly|might)\b", re.I)
_META = re.compile(r"\b(table\s*\d+|T\d+\b|page\s*\d+|\.html|\.png|table_id|annotation)\b", re.I)


def parse_generation(doc_id: str, stratum: str, raw: str, table_ids: List[str]) -> CandidateCondition:
    """Parse the JSON reply; malformed outputs are kept with a rejection reason."""
    try:
        d = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        labels = {str(table_ids[int(x["table_id"].lstrip("T")) - 1]): x["label"].strip().lower() for x in d["table_labels"]}
        evidence = {str(table_ids[int(x["table_id"].lstrip("T")) - 1]): x.get("evidence", "") for x in d["table_labels"]}
        return CandidateCondition(doc_id, stratum, d["condition"].strip(), labels, evidence)
    except Exception as e:  # noqa: BLE001
        return CandidateCondition(doc_id, stratum, "", {}, rejected=f"malformed: {e}")


def automatic_checks(cand: CandidateCondition, table_ids: List[str], seen: set) -> CandidateCondition:
    """Reject malformed, duplicate, ambiguous / under-specified, or label-inconsistent proposals."""
    if cand.rejected:
        return cand
    text = cand.condition
    if len(text.split()) < 4:
        cand.rejected = "under-specified"
    elif _VAGUE.search(text):
        cand.rejected = "ambiguous wording"
    elif _META.search(text):
        cand.rejected = "mentions identifiers or annotation fields"
    elif set(cand.labels) != {str(t) for t in table_ids} or any(v not in ("yes", "no") for v in cand.labels.values()):
        cand.rejected = "label inconsistency"
    elif text.lower() in seen:
        cand.rejected = "duplicate"
    seen.add(text.lower())
    return cand


def to_dataset_entry(cand: CandidateCondition, condition_id: str) -> dict:
    """Record kept in ``demo_per_doc_<dataset>.json``."""
    return {"condition_id": condition_id, "condition": cand.condition, "category": cand.stratum,
            "gold_yes_tables": [t for t, l in cand.labels.items() if l == "yes"]}


def generate_for_document(client, doc_id: str, tables: List[dict], stratum: str, seen: set) -> CandidateCondition:
    """One generation call (OpenAI-compatible ``client``) + automatic checks."""
    resp = client.chat.completions.create(model=GENERATOR_MODEL, temperature=0.7, messages=build_prompt(tables, stratum))
    ids = [str(t["table_id"]) for t in tables]
    return automatic_checks(parse_generation(doc_id, stratum, resp.choices[0].message.content or "", ids), ids, seen)


def stratified_sample(entries: List[dict], fraction: float = 0.05, seed: int = 0) -> List[dict]:
    """Stratified 5% sample handed to the human validators (Appendix A.1.6)."""
    rng = random.Random(seed)
    by: Dict[str, List[dict]] = {}
    for e in entries:
        by.setdefault(e.get("category", ""), []).append(e)
    out: List[dict] = []
    for cat, items in by.items():
        out += rng.sample(items, max(1, int(round(fraction * len(items)))))
    return out


def document_level_split(doc_ids: List[str], ratios=(0.8, 0.1, 0.1), seed: int = 0,
                         strata: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Train/dev/test assignment at the document level (Appendix A.1.2).

    FinConTab and PubConTab inherit the split of their source corpora, so this
    is only used for DegConTab, where ``strata`` maps each document to its
    controlled-attribute combination (header level, degradation, border style)
    and the ratios are applied within every stratum. No document appears in
    more than one split.
    """
    rng = random.Random(seed)
    groups: Dict[str, List[str]] = {}
    for d in sorted(doc_ids):
        groups.setdefault((strata or {}).get(d, ""), []).append(d)
    out: Dict[str, str] = {}
    for _, ids in sorted(groups.items()):
        rng.shuffle(ids)
        n = len(ids)
        n_train, n_dev = int(round(ratios[0] * n)), int(round(ratios[1] * n))
        for i, d in enumerate(ids):
            out[d] = "train" if i < n_train else ("dev" if i < n_train + n_dev else "test")
    return out
