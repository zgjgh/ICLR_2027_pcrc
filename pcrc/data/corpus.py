"""Corpus reader.

Every dataset directory follows the bundled layout::

    <corpus>/<dataset>/
        docs/<doc_id>/doc_meta.json          {doc_id, n_pages, pages:[{page_num, page_size}], tables:[{table_id, page_num, bbox, table_png_relpath, html?}]}
        docs/<doc_id>/tables/page_<P>_t<ID>.png   cropped table images
        docs/<doc_id>/table_html/*.html      reference HTML (optional; else inline in doc_meta)
        docs/<doc_id>/table_ocr/*.txt        per-table OCR text (optional; else the table block OCR)
        segment_json/<doc_id>_p<P>.json      {meta:{doc_id, page, page_size}, blocks:{key: {type, bbox, ocr, table_id?}}}
        partition/<doc_id>.json              relevance index I(D) written by experiments.build_index

Condition files (``data/demo_per_doc_<dataset>.json``) map a document to its
conditions and gold ``yes`` tables; ``data/demo_conditions_<dataset>.json``
lists the condition ids of the demo subset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image

from .. import config
from .schema import Condition, Document, Paragraph, RelevanceIndex, Table


# ----------------------------------------------------------------- blocks
def _page_of_key(key: str) -> int:
    prefix = key.split("__")[0]
    try:
        return int(prefix.rsplit("_p", 1)[1])
    except (IndexError, ValueError):
        return 0


def load_blocks(dataset: str, doc_id: str) -> Dict[str, dict]:
    """All segmentation blocks of one document, keyed by globally unique key."""
    seg_dir = config.corpus_dir(dataset) / "segment_json"
    blocks: Dict[str, dict] = {}
    for f in sorted(seg_dir.glob(f"{doc_id}_p*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for k, b in d["blocks"].items():
            blocks[k if "__" in k else f"{f.stem}__{k}"] = b
    return blocks


def reading_order(keys: Iterable[str], blocks: Dict[str, dict]) -> List[str]:
    return sorted(keys, key=lambda k: (_page_of_key(k), blocks.get(k, {}).get("bbox", [0, 0, 0, 0])[1]))


# --------------------------------------------------------------- document
def load_document(dataset: str, doc_id: str) -> Optional[Document]:
    root = config.corpus_dir(dataset)
    doc_dir = root / "docs" / doc_id
    meta_path = doc_dir / "doc_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    blocks = load_blocks(dataset, doc_id)

    paragraphs = [
        Paragraph(key=k, page=_page_of_key(k), bbox=b.get("bbox", [0, 0, 0, 0]),
                  text=b.get("ocr", ""), block_type=b.get("type", "paragraph"))
        for k, b in blocks.items() if b.get("type") != "table"
    ]
    page_sizes = {}
    for p in meta.get("pages", []):
        if isinstance(p, dict) and "page_num" in p:
            size = p.get("page_size") or [p.get("width", 612), p.get("height", 792)]
            page_sizes[int(p["page_num"])] = [float(size[0]), float(size[1])]

    tables: List[Table] = []
    for t in meta.get("tables", []):
        tid, page = t["table_id"], int(t["page_num"])
        stem = f"page_{page}_t{tid}"
        ocr_file = doc_dir / "table_ocr" / f"{stem}.txt"
        if ocr_file.exists():
            ocr = ocr_file.read_text(encoding="utf-8")
        else:
            ocr = next((b.get("ocr", "") for k, b in blocks.items()
                        if b.get("type") == "table" and str(b.get("table_id")) == str(tid)
                        and _page_of_key(k) == page), "")
        html_file = doc_dir / "table_html" / f"{stem}.html"
        html = html_file.read_text(encoding="utf-8") if html_file.exists() else t.get("html", "")
        table_key = next((k for k, b in blocks.items() if b.get("type") == "table"
                          and str(b.get("table_id")) == str(tid) and _page_of_key(k) == page),
                         f"{doc_id}_p{page}__table_{tid}")
        tables.append(Table(key=table_key, table_id=tid, page=page,
                            bbox=list(t.get("bbox", [0, 0, 0, 0])),
                            image_path=str(doc_dir / t.get("table_png_relpath", f"tables/{stem}.png")),
                            ocr_text=ocr, html_ref=html))
    return Document(doc_id=doc_id, n_pages=int(meta.get("n_pages", 0)),
                    page_sizes=page_sizes, paragraphs=paragraphs, tables=tables)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ------------------------------------------------------- relevance index
def index_path(dataset: str, doc_id: str) -> Path:
    return config.corpus_dir(dataset) / "partition" / f"{doc_id}.json"


def load_index(dataset: str, doc_id: str) -> Optional[RelevanceIndex]:
    p = index_path(dataset, doc_id)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return RelevanceIndex(
        doc_id=doc_id,
        shared_keys=list(d.get("shared_paragraph_keys", [])),
        exclusive_keys={t["table_key"]: list(t.get("exclusive_paragraph_keys", [])) for t in d.get("tables", [])},
        scores=d.get("scores", {}),
        doc_scores=d.get("doc_scores", {}),
    )


def save_index(dataset: str, doc: Document, index: RelevanceIndex) -> Path:
    p = index_path(dataset, doc.doc_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_id": doc.doc_id,
        "shared_paragraph_keys": index.shared_keys,
        "tables": [{"table_key": t.key, "table_id": t.table_id, "page": t.page, "bbox": t.bbox,
                    "exclusive_paragraph_keys": index.exclusive_keys.get(t.key, [])} for t in doc.tables],
        "scores": index.scores,
        "doc_scores": index.doc_scores,
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ------------------------------------------------------------ conditions
def load_conditions(dataset: str, subset_only: bool = False) -> Dict[str, List[Condition]]:
    """doc_id -> conditions. ``subset_only`` restricts to the demo subset ids."""
    d = json.loads(config.per_doc_conditions_file(dataset).read_text(encoding="utf-8"))
    keep = None
    if subset_only:
        keep = set(json.loads(config.demo_subset_ids_file(dataset).read_text(encoding="utf-8")))
    out: Dict[str, List[Condition]] = {}
    for doc_id, entry in d["docs"].items():
        conds = []
        for c in entry["conditions"]:
            if keep is not None and c["condition_id"] not in keep:
                continue
            conds.append(Condition(condition_id=c["condition_id"], text=c["condition"],
                                   category=c.get("category", ""), gold_yes_tables=c.get("gold_yes_tables", [])))
        if conds:
            out[doc_id] = conds
    return out


def iter_documents(dataset: str, subset_only: bool = False, limit: Optional[int] = None):
    """Yield (Document, RelevanceIndex) for every document with conditions."""
    conds = load_conditions(dataset, subset_only)
    for n, (doc_id, cs) in enumerate(sorted(conds.items())):
        if limit is not None and n >= limit:
            break
        doc = load_document(dataset, doc_id)
        if doc is None:
            continue
        doc.conditions = cs
        yield doc, load_index(dataset, doc_id)
