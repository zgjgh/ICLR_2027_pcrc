"""Shared helpers for the experiment entry points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from pcrc import config
from pcrc.data.corpus import load_image

# One-shot formatting demonstration per dataset (a real reference table + its HTML).
SAMPLE_TABLE = {
    "fintabnet": ("docs/HAL_2015/tables/page_75_t125183.png", "docs/HAL_2015/table_html/page_75_t125183.html"),
    "pubtables": ("docs/pubtables_v2_doc_001/tables/page_1_tPMC4829176_table_0.png",
                  "docs/pubtables_v2_doc_001/table_html/page_1_tPMC4829176_table_0.html"),
    "wild": ("docs/wild_doc_001/tables/page_1_timg_000161_t0.png", "docs/wild_doc_001/table_html/page_1_timg_000161_t0.html"),
}


def load_sample(dataset: str):
    root = config.corpus_dir(dataset)
    img_rel, html_rel = SAMPLE_TABLE[config.DATASET_ALIASES[dataset.lower()]]
    html_path = root / html_rel
    html = html_path.read_text(encoding="utf-8") if html_path.exists() else "<table><tr><td></td></tr></table>"
    return load_image(str(root / img_rel)), html


def dataset_argument(parser: argparse.ArgumentParser, multiple: bool = False):
    if multiple:
        parser.add_argument("--datasets", nargs="+", default=list(config.DATASETS), help="fincontab pubcontab degcontab")
    else:
        parser.add_argument("--dataset", required=True, choices=sorted(config.DATASET_ALIASES))
    parser.add_argument("--subset", action="store_true", help="restrict to the demo condition subset")
    parser.add_argument("--limit", type=int, default=None, help="number of documents (debugging)")
    parser.add_argument("--device", default="cuda:0")
    return parser


def subset_ids(dataset: str) -> Optional[set]:
    p = config.demo_subset_ids_file(dataset)
    return set(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else None


def ensure_index(dataset: str, doc, index, cfg=config.DEFAULT_SELECTION, mode: str = "pcrc", encoder=None):
    """Return the relevance index, building it with the given selection config when
    it is missing or when a non-default configuration is requested."""
    from pcrc.preprocessing.partition import build_index
    if index is not None and cfg == config.DEFAULT_SELECTION and mode == "pcrc":
        return index
    if encoder is None:
        from pcrc.preprocessing.embedding import NVEmbedEncoder
        encoder = NVEmbedEncoder()
    par = encoder.encode([p.text for p in doc.paragraphs])
    tab = encoder.encode([t.ocr_text for t in doc.tables])
    return build_index(doc, par, tab, cfg=cfg, mode=mode)
