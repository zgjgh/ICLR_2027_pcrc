"""Source-grounded condition construction (Section 4, Appendix A.1.5-A.1.7).

For every document, the reference HTML of all tables (page order) is given
to GPT-5.2 with one target stratum per call; proposals pass the automatic
checks, are stratified-sampled for human validation, and are written in
the ``demo_per_doc_<dataset>.json`` format.

    python -m experiments.build_conditions --dataset fincontab --per-doc 3 --out data/generated_fintabnet.json
"""
from __future__ import annotations

import argparse
import json

from openai import OpenAI

from pcrc.data.corpus import load_document
from pcrc.datasets.condition_generation import (generate_for_document, schedule_strata, stratified_sample,
                                                to_dataset_entry)

from experiments.common import dataset_argument


def main():
    ap = dataset_argument(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--doc-ids", nargs="*", default=None, help="documents to process (default: all in corpus/docs)")
    ap.add_argument("--per-doc", type=int, default=3, help="conditions per document (Table 4: three)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from pcrc import config
    doc_ids = a.doc_ids or sorted(p.name for p in (config.corpus_dir(a.dataset) / "docs").iterdir() if p.is_dir())
    client = OpenAI()
    seen, docs, rejected = set(), {}, []
    plan = schedule_strata(len(doc_ids) * a.per_doc)
    call = 0
    for doc_id in doc_ids:
        doc = load_document(a.dataset, doc_id)
        if doc is None:
            continue
        tables = [{"table_id": t.table_id, "html": t.html_ref} for t in sorted(doc.tables, key=lambda t: t.page)]
        kept = []
        for _ in range(a.per_doc):
            cand = generate_for_document(client, doc_id, tables, plan[call], seen)
            call += 1
            if cand.rejected:
                rejected.append({"doc_id": doc_id, "reason": cand.rejected})
                continue
            kept.append(to_dataset_entry(cand, f"{doc_id}_{cand.stratum}_{len(kept)}"))
        if kept:
            docs[doc_id] = {"conditions": kept}
    entries = [c for d in docs.values() for c in d["conditions"]]
    payload = {"docs": docs, "validation_sample": [c["condition_id"] for c in stratified_sample(entries)],
               "rejected": rejected}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"{len(entries)} conditions over {len(docs)} documents; {len(rejected)} rejected; wrote {a.out}")


if __name__ == "__main__":
    main()
