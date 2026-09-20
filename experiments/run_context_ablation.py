"""Effect of surrounding context on extraction (Appendix A.3.8, Table 14).

For each table, Claude Opus 4.5, GPT-5.2, and Qwen3-VL-Plus are queried
under two otherwise identical settings: Condition A gives the table image
and unstructured OCR; Condition B adds the shared-related paragraph and
the top-2 exclusive-related paragraphs. Both use the same one-shot HTML
example. TEDS (cell-count weighted) is reported per condition, and the
delta B - A.

    python -m experiments.run_context_ablation --dataset degcontab --backbone claude-opus-4.5
"""
from __future__ import annotations

import argparse
import json

from pcrc import config
from pcrc.data.corpus import iter_documents, load_image
from pcrc.evaluation.teds import cell_count, teds_score
from pcrc.inference.api_backbones import BACKBONES, APIBackbone

from experiments.common import dataset_argument, ensure_index, load_sample


def main():
    ap = dataset_argument(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--backbone", required=True, choices=list(BACKBONES))
    a = ap.parse_args()

    backbone = APIBackbone(a.backbone)
    sample_image, sample_html = load_sample(a.dataset)
    rows = []
    for doc, idx in iter_documents(a.dataset, subset_only=a.subset, limit=a.limit):
        idx = ensure_index(a.dataset, doc, idx)
        text_of = {p.key: p.text for p in doc.paragraphs}
        shared = [text_of[k] for k in idx.shared_keys[:1] if k in text_of]
        for t in doc.tables:
            excl = [text_of[k] for k in idx.exclusive_keys.get(t.key, [])[:2] if k in text_of]
            img = load_image(t.image_path)
            html_a = backbone.extract(img, t.ocr_text, sample_image, sample_html, paragraphs=None)
            html_b = backbone.extract(img, t.ocr_text, sample_image, sample_html, paragraphs=shared + excl)
            rows.append({"doc_id": doc.doc_id, "table_id": str(t.table_id), "cells": cell_count(t.html_ref),
                         "teds_a": teds_score(html_a, t.html_ref), "teds_b": teds_score(html_b, t.html_ref)})
    w = sum(r["cells"] for r in rows) or 1
    A = sum(r["cells"] * r["teds_a"] for r in rows) / w
    B = sum(r["cells"] * r["teds_b"] for r in rows) / w
    out = config.OUTPUT_ROOT / "scores" / f"context_ablation_{config.DATASET_ALIASES[a.dataset.lower()]}_{a.backbone}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backbone": a.backbone, "dataset": a.dataset, "teds_a": A, "teds_b": B,
                               "delta": B - A, "n_tables": len(rows), "rows": rows}, indent=1))
    print(f"{a.backbone} on {a.dataset}: A {A:.3f}  B {B:.3f}  delta {B - A:+.3f}")


if __name__ == "__main__":
    main()
