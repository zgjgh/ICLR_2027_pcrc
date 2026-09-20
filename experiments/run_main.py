"""Main comparison (Section 5.2 / 5.3; Tables 2, 3, 7).

Methods: ``qwen`` (single-shot), ``pcrc``, ``pcrc_nocache``, and the six
modular pipelines ``qwen+tatr qwen+mineru tatr+tapas tatr+qwen mineru+qwen
mineru+tapas``. Every method is evaluated on exactly the same documents,
conditions, and candidate tables, and consumes the same parsed inputs.

    python -m experiments.run_main --dataset fincontab --method pcrc
    python -m experiments.run_main --dataset fincontab --method qwen
    python -m experiments.run_main --dataset fincontab --method qwen+tatr --mineru-out outputs/mineru
"""
from __future__ import annotations

import argparse

from pcrc import config
from pcrc.baselines.modular import PIPELINES, ModularPipeline
from pcrc.data.corpus import iter_documents
from pcrc.evaluation.score import read_records, write_records
from pcrc.inference.engine import VLMEngine
from pcrc.inference.pcrc import PCRCOptions, PCRCRunner
from pcrc.inference.single_shot import SingleShotRunner

from experiments.common import dataset_argument, ensure_index, load_sample

METHODS = ["qwen", "pcrc", "pcrc_nocache"] + list(PIPELINES)


def build_modular(name: str, a, engine_factory):
    from pcrc.baselines.extractors import MinerUExtractor, TATRExtractor
    from pcrc.baselines.judges import QwenHTMLJudge, TAPASJudge
    kind, ext, judge = PIPELINES[name]
    extractor = TATRExtractor(device=a.device) if ext == "tatr" else MinerUExtractor(a.mineru_out)
    j = None
    if judge == "tapas":
        j = TAPASJudge(device=a.device)
    elif judge == "qwen":
        j = QwenHTMLJudge(engine_factory())
    base = {}
    if kind == "judge_then_extract":                      # reuse the single-shot Qwen decisions
        for r in read_records(a.dataset, "qwen"):
            base[(r["doc_id"], r["condition_id"], str(r["table_id"]))] = type("R", (), r)()
    return ModularPipeline(name, extractor, j, dataset=a.dataset, single_shot_outcomes=base)


def main():
    ap = dataset_argument(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--model", default=config.QWEN_MODEL_PATH, help="Qwen2.5-VL-7B (default) or GLM-4.5V")
    ap.add_argument("--mineru-out", default=str(config.OUTPUT_ROOT / "mineru"))
    a = ap.parse_args()

    sample_image, sample_html = load_sample(a.dataset)
    engine_factory = lambda: VLMEngine(a.model, device=a.device)  # noqa: E731
    method_tag = a.method if a.model == config.QWEN_MODEL_PATH else f"{a.method}@{a.model.split('/')[-1]}"

    if a.method == "qwen":
        runner = SingleShotRunner(engine_factory(), sample_image, sample_html, dataset=a.dataset, method=method_tag)
        run = lambda doc, idx: runner.run_document(doc)  # noqa: E731
    elif a.method in ("pcrc", "pcrc_nocache"):
        opts = PCRCOptions(reuse=(a.method == "pcrc"))
        runner = PCRCRunner(engine_factory(), sample_image, sample_html, opts, dataset=a.dataset, method=method_tag)
        run = lambda doc, idx: runner.run_document(doc, ensure_index(a.dataset, doc, idx))  # noqa: E731
    else:
        pipe = build_modular(a.method, a, engine_factory)
        run = lambda doc, idx: pipe.run_document(doc)  # noqa: E731

    first = True
    for doc, idx in iter_documents(a.dataset, subset_only=a.subset, limit=a.limit):
        write_records(a.dataset, method_tag, list(run(doc, idx)), append=not first)
        first = False
    print(f"done: {a.dataset} / {method_tag}")


if __name__ == "__main__":
    main()
