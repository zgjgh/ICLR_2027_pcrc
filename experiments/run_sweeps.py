"""Hyper-parameter sensitivity (Table 10) and decay-rate sweep (Table 12).

Each block varies one parameter while keeping the other three at their
defaults (K_shared = 4, tau_shared = 0.12, K_excl = 3, tau_excl = 0.3);
the lambda sweep replaces the spatial factor by exp(-lam * d_hat).

    python -m experiments.run_sweeps --dataset fincontab --param k_excl
    python -m experiments.run_sweeps --dataset fincontab --param lam
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from pcrc import config
from pcrc.data.corpus import iter_documents
from pcrc.evaluation.score import write_records
from pcrc.inference.engine import VLMEngine
from pcrc.inference.pcrc import PCRCOptions, PCRCRunner
from pcrc.preprocessing.embedding import NVEmbedEncoder

from experiments.common import dataset_argument, ensure_index, load_sample


def main():
    ap = dataset_argument(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--param", required=True, choices=list(config.SWEEP_GRID))
    ap.add_argument("--values", nargs="*", type=float, default=None, help="override the grid")
    a = ap.parse_args()

    values = a.values if a.values else config.SWEEP_GRID[a.param]
    sample_image, sample_html = load_sample(a.dataset)
    engine = VLMEngine(device=a.device)
    encoder = NVEmbedEncoder(device=a.device)
    for v in values:
        v = int(v) if a.param.startswith("k_") else float(v)
        cfg = replace(config.DEFAULT_SELECTION, **{a.param: v})
        tag = f"sweep_{a.param}={v}"
        runner = PCRCRunner(engine, sample_image, sample_html, PCRCOptions(), dataset=a.dataset, method=tag)
        first = True
        for doc, _ in iter_documents(a.dataset, subset_only=a.subset, limit=a.limit):
            idx = ensure_index(a.dataset, doc, None, cfg=cfg, encoder=encoder)
            write_records(a.dataset, tag, list(runner.run_document(doc, idx)), append=not first)
            first = False
        print(f"done: {a.dataset} / {tag}")


if __name__ == "__main__":
    main()
