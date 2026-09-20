"""The six modular pipelines of Section 5.1 under the X+Y convention
(X runs first, Y consumes its output).

Judge-then-extract (keep the single-shot Qwen judgment, change the extractor):
    Qwen+TATR, Qwen+MinerU   a table counts as ``yes`` when single-shot Qwen
                             returned HTML rather than ``no``; TATR or MinerU
                             then extracts the accepted tables, so these two
                             pipelines differ from Qwen only in the extractor.

Extract-then-judge (extract every candidate first, then judge):
    TATR+TAPAS, TATR+Qwen, MinerU+Qwen, MinerU+TAPAS
                             TATR or MinerU extracts HTML for every candidate
                             table; the downstream judge (TAPAS on the
                             structured table, or Qwen on image + HTML)
                             decides whether the extracted table satisfies
                             the condition.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from ..data.corpus import load_image
from ..data.schema import Condition, Document, TableOutcome

PIPELINES = {
    "qwen+tatr": ("judge_then_extract", "tatr", None),
    "qwen+mineru": ("judge_then_extract", "mineru", None),
    "tatr+tapas": ("extract_then_judge", "tatr", "tapas"),
    "tatr+qwen": ("extract_then_judge", "tatr", "qwen"),
    "mineru+qwen": ("extract_then_judge", "mineru", "qwen"),
    "mineru+tapas": ("extract_then_judge", "mineru", "tapas"),
}


def _extract(extractor, table, word_boxes=None) -> str:
    if hasattr(extractor, "weights"):                       # TATR needs the crop + word boxes
        return extractor.extract(load_image(table.image_path), word_boxes)
    return extractor.extract(table.image_path)              # MinerU reads its batch output


class ModularPipeline:
    def __init__(self, name: str, extractor, judge=None, dataset: str = "",
                 single_shot_outcomes: Optional[Dict[tuple, TableOutcome]] = None):
        self.name = name
        self.kind, _, self.judge_kind = PIPELINES[name]
        self.extractor = extractor
        self.judge = judge
        self.dataset = dataset
        self.single_shot = single_shot_outcomes or {}   # (doc_id, condition_id, table_id) -> outcome

    def run_condition(self, doc: Document, cond: Condition, word_boxes: Optional[Dict[str, List[dict]]] = None) -> List[TableOutcome]:
        outs = []
        for table in doc.tables:
            key = (doc.doc_id, cond.condition_id, str(table.table_id))
            if self.kind == "judge_then_extract":
                base = self.single_shot.get(key)
                predicted = base.predicted if base is not None else "no"
                html = _extract(self.extractor, table, (word_boxes or {}).get(table.key)) if predicted == "yes" else None
                answers = [predicted]
            else:
                html_all = _extract(self.extractor, table, (word_boxes or {}).get(table.key))
                if self.judge_kind == "tapas":
                    predicted = self.judge.judge(cond.text, html_all)
                else:
                    predicted = self.judge.judge(cond.text, load_image(table.image_path), html_all)
                html = html_all if predicted == "yes" else None
                answers = [predicted]
            outs.append(TableOutcome(dataset=self.dataset, method=self.name, doc_id=doc.doc_id,
                                     condition_id=cond.condition_id, category=cond.category, table_id=table.table_id,
                                     gold=cond.label(table.table_id), predicted=predicted, html_pred=html, rounds=1,
                                     answers=answers, prefill_actual=0, prefill_nocache=0, decode=0))
        return outs

    def run_document(self, doc: Document, word_boxes=None) -> Iterable[TableOutcome]:
        for cond in doc.conditions:
            yield from self.run_condition(doc, cond, word_boxes)
