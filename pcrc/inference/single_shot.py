"""Single-shot Qwen baseline (Section 5.1 (a), Appendix A.2.1).

One decoding pass per (condition, table): the model receives the table
image, table OCR, condition, and a one-shot sample table image with its
reference HTML, decides whether the table satisfies the condition and, if
so, extracts it, returning either ``no`` or the table HTML. There is no
cache to reuse, so prefill_actual == prefill_nocache for every call.

The same routine drives GLM-4.5V single-shot (Appendix A.3.7) by loading
the engine with ``GLM_MODEL_PATH``.
"""
from __future__ import annotations

from typing import Iterable, List

from .. import config
from ..data.corpus import load_image
from ..data.schema import Condition, Document, TableOutcome
from . import prompts as P
from .engine import VLMEngine


class SingleShotRunner:
    def __init__(self, engine: VLMEngine, sample_image, sample_html: str, dataset: str = "", method: str = "qwen"):
        self.engine = engine
        self.sample_image = sample_image
        self.sample_html = sample_html
        self.dataset = dataset
        self.method = method

    def run_condition(self, doc: Document, cond: Condition) -> List[TableOutcome]:
        acc = self.engine.accountant
        outcomes = []
        for table in doc.tables:
            acc.reset()
            msgs = P.single_shot_messages(cond.text, load_image(table.image_path), table.ocr_text,
                                          self.sample_image, self.sample_html)
            raw = self.engine.generate_once(msgs, config.MAX_NEW_TOKENS_EXTRACT, label="single_shot")
            html = P.parse_single_shot(raw)
            t = acc.totals()
            outcomes.append(TableOutcome(
                dataset=self.dataset, method=self.method, doc_id=doc.doc_id, condition_id=cond.condition_id,
                category=cond.category, table_id=table.table_id, gold=cond.label(table.table_id),
                predicted="yes" if html else "no", html_pred=html, rounds=1,
                answers=["yes" if html else "no"],
                prefill_actual=t["prefill_actual"], prefill_nocache=t["prefill_nocache"], decode=t["decode"]))
        return outcomes

    def run_document(self, doc: Document) -> Iterable[TableOutcome]:
        for cond in doc.conditions:
            yield from self.run_condition(doc, cond)
