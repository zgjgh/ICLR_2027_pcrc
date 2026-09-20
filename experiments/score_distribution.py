"""Empirical distribution of the two factors of the relevance score
(Appendix A.3.6, Table 11) and the spread of the spatial factor under the
decay-rate sweep (first block of Table 12).

Spread is the ratio q95/q05, the multiplicative gap a factor can open
between two candidate paragraphs.

    python -m experiments.score_distribution --datasets fincontab pubcontab degcontab
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from pcrc import config
from pcrc.data.corpus import iter_documents
from pcrc.preprocessing.embedding import CachedEncoder, NVEmbedEncoder, semantic_similarity
from pcrc.preprocessing.relevance import factor_quantiles, spatial_proximity

from experiments.common import dataset_argument


def main():
    ap = dataset_argument(argparse.ArgumentParser(description=__doc__), multiple=True)
    a = ap.parse_args()
    encoder = NVEmbedEncoder(device=a.device)
    report = {}
    for ds in a.datasets:
        enc = CachedEncoder(encoder, config.corpus_dir(ds) / "embeddings.json")
        sem, spa = [], {lam: [] for lam in config.SWEEP_GRID["lam"]}
        for doc, _ in iter_documents(ds, subset_only=a.subset, limit=a.limit):
            if not doc.paragraphs or not doc.tables:
                continue
            par = enc.encode([p.key for p in doc.paragraphs], [p.text for p in doc.paragraphs])
            tab = enc.encode([t.key for t in doc.tables], [t.ocr_text for t in doc.tables])
            sem.append(semantic_similarity(par, tab).ravel())
            for lam in spa:
                spa[lam].append(spatial_proximity(doc, lam=lam).ravel())
        report[ds] = {"semantic": factor_quantiles(np.concatenate(sem)),
                      "spatial": {str(lam): factor_quantiles(np.concatenate(v)) for lam, v in spa.items()}}
        print(ds, "semantic spread %.2fx" % report[ds]["semantic"]["spread"],
              "spatial spread(lam=1) %.2fx" % report[ds]["spatial"]["1.0"]["spread"])
    out = config.OUTPUT_ROOT / "scores" / "score_distribution.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
