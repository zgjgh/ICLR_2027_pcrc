"""Backbone generality (Appendix A.3.7, Table 13).

* Frontier VLMs (Claude Opus 4.5, GPT-5.2, Qwen3-VL-Plus) as single-shot
  baselines under the Appendix A.2.1 protocol (API endpoints; no cache-level
  cost is measurable).
* GLM-4.5V, an open-weight backbone whose cache is accessible: single-shot,
  full PCRC, and PCRC w/o cache, through the same engine as Qwen2.5-VL-7B.

    python -m experiments.run_backbones --dataset fincontab --backbone gpt-5.2
    python -m experiments.run_backbones --dataset fincontab --backbone glm-4.5v --mode pcrc
"""
from __future__ import annotations

import argparse

from pcrc import config
from pcrc.data.corpus import iter_documents, load_image
from pcrc.data.schema import TableOutcome
from pcrc.evaluation.score import write_records
from pcrc.inference.api_backbones import BACKBONES, APIBackbone

from experiments.common import dataset_argument, ensure_index, load_sample


def run_api_single_shot(backbone: APIBackbone, dataset: str, doc, sample_image, sample_html):
    for cond in doc.conditions:
        for table in doc.tables:
            html = backbone.single_shot(cond.text, load_image(table.image_path), table.ocr_text, sample_image, sample_html)
            yield TableOutcome(dataset=dataset, method=f"single_shot@{backbone.name}", doc_id=doc.doc_id,
                               condition_id=cond.condition_id, category=cond.category, table_id=table.table_id,
                               gold=cond.label(table.table_id), predicted="yes" if html else "no", html_pred=html,
                               rounds=1, answers=["yes" if html else "no"], prefill_actual=0, prefill_nocache=0, decode=0)


def main():
    ap = dataset_argument(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--backbone", required=True, choices=list(BACKBONES) + ["glm-4.5v"])
    ap.add_argument("--mode", choices=["single_shot", "pcrc", "pcrc_nocache"], default="single_shot")
    a = ap.parse_args()

    sample_image, sample_html = load_sample(a.dataset)
    if a.backbone == "glm-4.5v":
        from pcrc.inference.engine import VLMEngine
        from pcrc.inference.pcrc import PCRCOptions, PCRCRunner
        from pcrc.inference.single_shot import SingleShotRunner
        engine = VLMEngine(config.GLM_MODEL_PATH, device=a.device)
        tag = f"{a.mode}@glm-4.5v"
        if a.mode == "single_shot":
            runner = SingleShotRunner(engine, sample_image, sample_html, dataset=a.dataset, method=tag)
            run = lambda doc, idx: runner.run_document(doc)  # noqa: E731
        else:
            runner = PCRCRunner(engine, sample_image, sample_html, PCRCOptions(reuse=(a.mode == "pcrc")),
                                dataset=a.dataset, method=tag)
            run = lambda doc, idx: runner.run_document(doc, ensure_index(a.dataset, doc, idx))  # noqa: E731
    else:
        backbone = APIBackbone(a.backbone)
        tag = f"single_shot@{a.backbone}"
        run = lambda doc, idx: run_api_single_shot(backbone, a.dataset, doc, sample_image, sample_html)  # noqa: E731

    first = True
    for doc, idx in iter_documents(a.dataset, subset_only=a.subset, limit=a.limit):
        write_records(a.dataset, tag, list(run(doc, idx)), append=not first)
        first = False
    print(f"done: {a.dataset} / {tag}")


if __name__ == "__main__":
    main()
