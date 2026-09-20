"""PCRC preprocessing: build the table-centered relevance index I(D) for every
document of a dataset (Section 3.1) and write ``partition/<doc_id>.json``.

    python -m experiments.build_index --dataset fincontab
    python -m experiments.build_index --dataset fincontab --k-excl 2 --tau-shared 0.06 --lam 2.0
    python -m experiments.build_index --dataset fincontab --mode random     # Table 9 variants
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from pcrc import config
from pcrc.data.corpus import iter_documents, save_index
from pcrc.preprocessing.embedding import CachedEncoder, NVEmbedEncoder
from pcrc.preprocessing.partition import build_index

from experiments.common import dataset_argument


def main():
    ap = dataset_argument(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--k-shared", type=int, default=config.DEFAULT_SELECTION.k_shared)
    ap.add_argument("--tau-shared", type=float, default=config.DEFAULT_SELECTION.tau_shared)
    ap.add_argument("--k-excl", type=int, default=config.DEFAULT_SELECTION.k_excl)
    ap.add_argument("--tau-excl", type=float, default=config.DEFAULT_SELECTION.tau_excl)
    ap.add_argument("--lam", type=float, default=config.DEFAULT_SELECTION.lam)
    ap.add_argument("--relevance", choices=["product", "semantic_only", "spatial_only"], default="product")
    ap.add_argument("--mode", choices=["pcrc", "random", "all"], default="pcrc")
    a = ap.parse_args()

    cfg = replace(config.DEFAULT_SELECTION, k_shared=a.k_shared, tau_shared=a.tau_shared, k_excl=a.k_excl,
                  tau_excl=a.tau_excl, lam=a.lam, relevance=a.relevance)
    enc = CachedEncoder(NVEmbedEncoder(device=a.device), config.corpus_dir(a.dataset) / "embeddings.json")
    n = 0
    for doc, _ in iter_documents(a.dataset, subset_only=a.subset, limit=a.limit):
        par = enc.encode([p.key for p in doc.paragraphs], [p.text for p in doc.paragraphs])
        tab = enc.encode([t.key for t in doc.tables], [t.ocr_text for t in doc.tables])
        save_index(a.dataset, doc, build_index(doc, par, tab, cfg=cfg, mode=a.mode))
        n += 1
    enc.flush()
    print(f"indexed {n} documents of {a.dataset} with {cfg} mode={a.mode}")


if __name__ == "__main__":
    main()
