"""PCRC inference for a document-condition pair (Section 3.2, Algorithm 1).

    Require: shared cache C_head, tables T(D), exclusive sets E(T_i)
    for each table T_i in T(D):
        C <- C_head (+) [image(T_i); ocr(T_i)];  (a, c) <- Judge(C);  k <- 1
        while c != RELIABLE and k <= |E(T_i)|:
            drop the unreliable kv states;  C <- C (+) e_k;  (a, c) <- Judge(C);  k <- k + 1
        if c != RELIABLE:  a <- Judge(C (+) forced yes/no prompt)
        if a = yes: output Extract(C (+) extraction inputs)  else skip T_i
        discard the table-specific tail

The reliability signal is produced by the model itself: the judging prompt
permits ``uncertain`` alongside ``yes`` and ``no``; a definite answer is
RELIABLE. ``PCRCRunner`` also realises the Appendix A.3.4 / A.3.5 variants:

    gating=False        No reliability gating: every exclusive paragraph is
                        appended unconditionally (no early stopping), one
                        judgment per round, the last answer counts.
    single_judgment     All selected evidence in one call, judged once
                        (the "single judgment" protocol of Appendix A.3.5).
    reuse=False         PCRC w/o cache: identical schedule, full-prefix
                        recomputation at every call.
    evidence            "pcrc" | "shared_only" | "exclusive_only" restricts
                        the evidence types (Table 9); "random" / "all" are
                        produced upstream by pcrc.preprocessing.partition.

Token accounting: the head prefill is registered before the first
judgment call of the first table, so per-condition sums of the outcome
records are exact; in no-cache mode every call re-encodes its full prefix.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .. import config
from ..data.corpus import load_image
from ..data.schema import Condition, Document, RelevanceIndex, Table, TableOutcome
from ..preprocessing.partition import restrict_index
from . import prompts as P
from .engine import VLMEngine


@dataclass
class PCRCOptions:
    reuse: bool = True                 # kv-cache reuse across tables and rounds
    gating: bool = True                # reliability gate (uncertain -> append the next paragraph)
    single_judgment: bool = False      # all evidence at once, one judgment call
    evidence: str = "pcrc"             # pcrc | shared_only | exclusive_only
    max_new_tokens_judge: int = config.MAX_NEW_TOKENS_JUDGE
    max_new_tokens_extract: int = config.MAX_NEW_TOKENS_EXTRACT

    @property
    def method_name(self) -> str:
        if self.single_judgment:
            return "single_judgment"
        if not self.gating:
            return "no_gating"
        if self.evidence != "pcrc":
            return self.evidence
        return "pcrc" if self.reuse else "pcrc_nocache"


class PCRCRunner:
    def __init__(self, engine: VLMEngine, sample_image, sample_html: str,
                 options: PCRCOptions = PCRCOptions(), dataset: str = "", method: str = ""):
        self.engine = engine
        self.sample_image = sample_image
        self.sample_html = sample_html
        self.opt = options
        self.dataset = dataset
        self.method = method or options.method_name

    # ---------------------------------------------------------- one table
    def _judge(self, frame, label: str, forced: bool = False) -> str:
        raw = frame.generate(label, self.opt.max_new_tokens_judge, request=P.pcrc_judge_request(forced=forced))
        return P.parse_judgment(raw, allow_uncertain=not forced)

    def run_table(self, frame, table: Table, exclusive: List[str], doc: Document, cond: Condition) -> TableOutcome:
        acc = self.engine.accountant
        first_call = len(acc.calls)
        answers: List[str] = []

        frame.append(P.pcrc_tail(load_image(table.image_path), table.ocr_text))   # tail: image + OCR

        if self.opt.single_judgment:                                             # A.3.5 single-judgment protocol
            for k, e in enumerate(exclusive, 1):
                frame.append(P.pcrc_exclusive(e, k))
            answers.append(self._judge(frame, "judge_1", forced=True))
        else:
            a = self._judge(frame, "judge_1")
            answers.append(a)
            k = 1
            while (a == "uncertain" or not self.opt.gating) and k <= len(exclusive):
                frame.append(P.pcrc_exclusive(exclusive[k - 1], k))               # C <- C (+) e_k
                a = self._judge(frame, f"judge_{k + 1}")
                answers.append(a)
                k += 1
            if a == "uncertain":                                                  # exclusive set exhausted
                a = self._judge(frame, "forced", forced=True)
                answers.append(a)

        predicted = "yes" if answers[-1] == "yes" else "no"
        html = None
        if predicted == "yes":                                                    # extraction on the same cache
            frame.append(P.pcrc_extraction_turn(self.sample_image, self.sample_html))
            html = P.parse_html(frame.generate("extract", self.opt.max_new_tokens_extract))
        frame.discard_tail()

        calls = acc.calls[first_call:]
        return TableOutcome(dataset=self.dataset, method=self.method, doc_id=doc.doc_id,
                           condition_id=cond.condition_id, category=cond.category, table_id=table.table_id,
                           gold=cond.label(table.table_id), predicted=predicted, html_pred=html,
                           rounds=len(answers), answers=answers,
                           prefill_actual=sum(c.prefill_actual for c in calls),
                           prefill_nocache=sum(c.prefill_nocache for c in calls),
                           decode=sum(c.decode for c in calls))

    # ------------------------------------------------- one (D, q) pair
    def run_condition(self, doc: Document, index: RelevanceIndex, cond: Condition) -> List[TableOutcome]:
        index = restrict_index(index, self.opt.evidence)
        text_of = {p.key: p.text for p in doc.paragraphs}
        shared = [text_of[k] for k in index.shared_keys if k in text_of]

        self.engine.accountant.reset()
        frame = self.engine.new_frame(P.PCRC_SYSTEM, reuse=self.opt.reuse)
        frame.prefill_head(P.pcrc_head(cond.text, shared))          # C_head, encoded once per (D, q)
        outcomes = []
        for table in doc.tables:
            exclusive = [text_of[k] for k in index.exclusive_keys.get(table.key, []) if k in text_of]
            outcomes.append(self.run_table(frame, table, exclusive, doc, cond))
        frame.close()
        return outcomes

    def run_document(self, doc: Document, index: RelevanceIndex) -> Iterable[TableOutcome]:
        for cond in doc.conditions:
            yield from self.run_condition(doc, index, cond)
