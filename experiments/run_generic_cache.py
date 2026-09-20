"""Comparison with generic prefix caching (Appendix A.3.5; cache-management
block of Table 9 and the extra row of Table 3).

Serve Qwen2.5-VL-7B with vLLM (prefix caching and multimodal block hashing
enabled, block size 16), e.g.

    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --enable-prefix-caching --port 8000

then replay the PCRC schedule, or run the single-judgment protocol:

    python -m experiments.run_generic_cache --dataset fincontab --protocol pcrc_schedule
    python -m experiments.run_generic_cache --dataset fincontab --protocol single_judgment

Every request logs prompt tokens and cached tokens; the per-request log is
written next to the records for auditing.
"""
from __future__ import annotations

import argparse
import json

from pcrc import config
from pcrc.data.corpus import iter_documents
from pcrc.evaluation.score import records_dir, write_records
from pcrc.inference.generic_cache import GenericCacheClient, GenericCacheRunner

from experiments.common import dataset_argument, ensure_index, load_sample


def main():
    ap = dataset_argument(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--protocol", choices=["pcrc_schedule", "single_judgment"], default="pcrc_schedule")
    ap.add_argument("--api-base", default=config.VLLM_API_BASE)
    ap.add_argument("--model", default=config.QWEN_MODEL_PATH)
    a = ap.parse_args()

    sample_image, sample_html = load_sample(a.dataset)
    client = GenericCacheClient(model=a.model, api_base=a.api_base)
    runner = GenericCacheRunner(client, sample_image, sample_html, dataset=a.dataset,
                                single_judgment=(a.protocol == "single_judgment"))
    first = True
    for doc, idx in iter_documents(a.dataset, subset_only=a.subset, limit=a.limit):
        idx = ensure_index(a.dataset, doc, idx)
        write_records(a.dataset, runner.method, list(runner.run_document(doc, idx)), append=not first)
        first = False
    log = records_dir(a.dataset) / f"{runner.method}.requests.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in client.log), encoding="utf-8")
    print(f"done: {a.dataset} / {runner.method}; per-request log at {log}")


if __name__ == "__main__":
    main()
