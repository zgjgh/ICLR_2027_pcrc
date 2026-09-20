"""Score recorded runs into the paper's tables.

    python -m experiments.score --datasets fincontab pubcontab degcontab
    python -m experiments.score --datasets fincontab --methods qwen pcrc pcrc_nocache --subset

Prints, per dataset and method, judgment P/R/F1 (Table 7), composite
P_c/R_c/F1 (Table 2), and cost (Table 3); writes JSON with the per-category
breakdown (Table 8) under ``outputs/scores``.
"""
from __future__ import annotations

import argparse

from pcrc.evaluation.metrics import average_over_datasets, QualityScores
from pcrc.evaluation.score import format_row, save_scores, score_dataset

from experiments.common import dataset_argument, subset_ids


def main():
    ap = dataset_argument(argparse.ArgumentParser(description=__doc__), multiple=True)
    ap.add_argument("--methods", nargs="*", default=None)
    a = ap.parse_args()

    per_dataset_cats = {}
    for ds in a.datasets:
        scores = score_dataset(ds, a.methods, subset_ids(ds) if a.subset else None)
        print(f"\n== {ds} ==")
        for m, s in scores.items():
            print(format_row(m, s))
        save_scores(ds, scores, tag="subset" if a.subset else "full")
        for m, s in scores.items():
            per_dataset_cats.setdefault(m, []).append({c: QualityScores(**v) for c, v in s["per_category"].items()})
    # Table 8: per-stratum scores averaged over the datasets scored
    for m, cats in per_dataset_cats.items():
        if len(cats) == len(a.datasets):
            avg = average_over_datasets(cats)
            print(f"\nper-category ({m}, averaged over {len(cats)} datasets): " +
                  "  ".join(f"{c}: {v['f1']:.3f}/{v['f1_c']:.3f}" for c, v in avg.items()))


if __name__ == "__main__":
    main()
