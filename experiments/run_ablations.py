"""Ablation of context selection and early stopping (Appendix A.3.4, Table 9,
Figure 3).

Variants (Qwen backbone; retrieval variants budget-matched to K_shared + K_excl):
    random          paragraphs drawn at random, delivered in a single round with no cache reuse
    all             every paragraph of the document in the head, tail = image + OCR
    semantic_only   r = s_sem            spatial_only    r = s_spa
    shared_only     only S(D)            exclusive_only  only E(T_i)
    no_gating       every exclusive paragraph appended unconditionally

The generic prefix-cache rows of Appendix A.3.5 come from ``run_generic_cache``.

    python -m experiments.run_ablations --dataset fincontab --variant no_gating
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from pcrc import config
from pcrc.data.corpus import iter_documents
from pcrc.evaluation.score import write_records
from pcrc.inference.engine import VLMEngine
from pcrc.inference.pcrc import PCRCOptions, PCRCRunner

from experiments.common import dataset_argument, ensure_index, load_sample

VARIANTS = ["random", "all", "semantic_only", "spatial_only", "shared_only", "exclusive_only", "no_gating"]


def options_for(variant: str):
    """(index mode, selection config, PCRCOptions) for one variant."""
    cfg = config.DEFAULT_SELECTION
    mode, opts = "pcrc", PCRCOptions()
    if variant == "random":
        mode, opts = "random", PCRCOptions(single_judgment=True, reuse=False)   # one round, no cache reuse
    elif variant == "all":
        mode = "all"
    elif variant in ("semantic_only", "spatial_only"):
        cfg = replace(cfg, relevance=variant)
    elif variant in ("shared_only", "exclusive_only"):
        opts = PCRCOptions(evidence=variant)
    elif variant == "no_gating":
        opts = PCRCOptions(gating=False)
    return mode, cfg, opts


def main():
    ap = dataset_argument(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    a = ap.parse_args()

    mode, cfg, opts = options_for(a.variant)
    sample_image, sample_html = load_sample(a.dataset)
    runner = PCRCRunner(VLMEngine(device=a.device), sample_image, sample_html, opts, dataset=a.dataset, method=a.variant)
    from pcrc.preprocessing.embedding import NVEmbedEncoder
    encoder = NVEmbedEncoder(device=a.device)
    first = True
    for doc, idx in iter_documents(a.dataset, subset_only=a.subset, limit=a.limit):
        idx = ensure_index(a.dataset, doc, idx, cfg=cfg, mode=mode, encoder=encoder)
        write_records(a.dataset, a.variant, list(runner.run_document(doc, idx)), append=not first)
        first = False
    print(f"done: {a.dataset} / {a.variant}")


if __name__ == "__main__":
    main()
