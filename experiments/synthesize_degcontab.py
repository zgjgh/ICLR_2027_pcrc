"""DegConTab document synthesis (Appendix A.1.3-A.1.4).

Generates a multi-page document from a topic and table-structure templates,
renders it with the requested border style and visual degradation, records
the artefact boxes, and writes the corpus layout read by ``pcrc.data.corpus``
(``docs/<doc_id>/{doc_meta.json, pages/, tables/, table_html/, table_ocr/}``
and ``segment_json/``) with the text layer masked under the artefacts.

    python -m experiments.synthesize_degcontab --topic "quarterly logistics report" \
        --templates flat_2col nested_header_3level --out corpus/wild --doc-id wild_doc_900
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from openai import OpenAI

from pcrc.datasets.degcontab_synthesis import crop_tables, generate_document, render_document, sample_controls
from pcrc.preprocessing.parsing import mask_degraded_text, words_to_text, write_segment_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--templates", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="dataset directory, e.g. corpus/wild")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    controls = sample_controls(rng)
    doc = generate_document(OpenAI(), a.model, a.topic, a.templates, controls)
    out = Path(a.out)
    doc_dir = out / "docs" / a.doc_id
    pages = render_document(doc, controls, doc_dir / "pages", seed=a.seed)
    table_records = crop_tables(pages, doc_dir)

    # Text layer per page (the renderer's word boxes), masked under the artefacts;
    # the OCR text of a table block is what survives the masking, characters only.
    (doc_dir / "table_ocr").mkdir(parents=True, exist_ok=True)
    for p in pages:
        words = mask_degraded_text(p.words, p.artifacts)
        blocks = {}
        for j, b in enumerate(p.blocks):
            own = [w for w in words if w["block"] == j]
            rec = {"type": b["type"], "bbox": b["bbox"], "ocr": words_to_text(own)}
            if b["type"] == "table":
                rec["table_id"] = b["table_id"]
                (doc_dir / "table_ocr" / f"page_{p.page_index}_t{b['table_id']}.txt").write_text(rec["ocr"], encoding="utf-8")
            blocks[f"{b['type']}_{j}"] = rec
        write_segment_json(out / "segment_json", a.doc_id, p.page_index, blocks, p.size)

    meta = {"doc_id": a.doc_id, "n_pages": len(pages), "render_controls": controls.__dict__,
            "pages": [{"page_num": p.page_index, "page_size": list(p.size)} for p in pages],
            "tables": table_records}
    (doc_dir / "doc_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {a.doc_id}: {len(pages)} pages, {len(table_records)} tables, controls={controls}")


if __name__ == "__main__":
    main()
