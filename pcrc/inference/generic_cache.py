"""Comparison with generic prefix caching (Appendix A.3.5).

Two protocols are served through a vLLM server with automatic prefix
caching and multimodal block hashing enabled (block size 16):

* ``PCRC schedule + generic prefix cache`` replays PCRC's exact call
  sequence (same prompts, rounds, outputs) as ordinary chat requests; the
  engine decides what to keep, so quality is identical by construction and
  only the prefill actually computed changes.
* ``Single judgment + generic prefix cache`` supplies the shared and
  exclusive paragraphs in one call, judges once, extracts accepted tables,
  and relies on the generic cache to reuse the shared head across tables.

For every request we record ``usage.prompt_tokens`` and the cached tokens
reported by the server (``usage.prompt_tokens_details.cached_tokens``); the
prefill actually computed is their difference, and R.R. is taken against
the same call sequence with caching disabled (prefill_nocache = prompt
tokens), as in Table 3.

Every request is serialised with the head blocks first (identical text
across the tables of a document-condition pair), then the tail, so that the
generic cache can match the longest common prefix; judgment answers are not
carried into the next request (Algorithm 1 drops them).
"""
from __future__ import annotations

import base64
import io
from typing import Iterable, List, Optional

from .. import config
from ..data.corpus import load_image
from ..data.schema import Condition, Document, RelevanceIndex, Table, TableOutcome
from ..preprocessing.partition import restrict_index
from . import prompts as P


def _image_url(image) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def _serialise(segments: List[P.Segment]) -> List[dict]:
    """Segments -> OpenAI chat content parts (text and images interleaved in order)."""
    parts: List[dict] = []
    for seg in segments:
        for item in seg.content():
            parts.append(_image_url(item["image"]) if item["type"] == "image" else {"type": "text", "text": item["text"]})
    return parts


class GenericCacheClient:
    """OpenAI-compatible client for a vLLM server (``VLLM_API_BASE``)."""

    def __init__(self, model: str = config.QWEN_MODEL_PATH, api_base: str = config.VLLM_API_BASE, api_key: str = "EMPTY"):
        from openai import OpenAI

        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.log: List[dict] = []   # per-request records (prompt / cached / computed tokens)

    def complete(self, system: str, segments: List[P.Segment], max_tokens: int, label: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, temperature=0.0, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": _serialise(segments)}],
        )
        usage = resp.usage
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0)
        self.log.append({"label": label, "prompt_tokens": int(usage.prompt_tokens), "cached_tokens": cached,
                         "computed_tokens": int(usage.prompt_tokens) - cached,
                         "decode_tokens": int(usage.completion_tokens)})
        return resp.choices[0].message.content or ""


class GenericCacheRunner:
    """Replay the PCRC schedule (or the single-judgment protocol) through the generic cache."""

    def __init__(self, client: GenericCacheClient, sample_image, sample_html: str,
                 dataset: str = "", single_judgment: bool = False, gating: bool = True):
        self.client = client
        self.sample_image = sample_image
        self.sample_html = sample_html
        self.dataset = dataset
        self.single_judgment = single_judgment
        self.gating = gating
        self.method = "single_judgment_generic_cache" if single_judgment else "pcrc_generic_cache"

    def _judge(self, head, tail, forced=False, label="judge"):
        raw = self.client.complete(P.PCRC_SYSTEM, head + tail + [P.pcrc_judge_request(forced)], config.MAX_NEW_TOKENS_JUDGE, label)
        return P.parse_judgment(raw, allow_uncertain=not forced)

    def run_table(self, head: List[P.Segment], table: Table, exclusive: List[str], doc: Document, cond: Condition) -> TableOutcome:
        first = len(self.client.log)
        tail = [P.pcrc_tail(load_image(table.image_path), table.ocr_text)]
        answers: List[str] = []
        if self.single_judgment:
            tail += [P.pcrc_exclusive(e, k) for k, e in enumerate(exclusive, 1)]
            answers.append(self._judge(head, tail, forced=True, label="judge_1"))
        else:
            a = self._judge(head, tail, label="judge_1")
            answers.append(a)
            k = 1
            while (a == "uncertain" or not self.gating) and k <= len(exclusive):
                tail.append(P.pcrc_exclusive(exclusive[k - 1], k))
                a = self._judge(head, tail, label=f"judge_{k + 1}")
                answers.append(a)
                k += 1
            if a == "uncertain":
                a = self._judge(head, tail, forced=True, label="forced")
                answers.append(a)
        predicted = "yes" if answers[-1] == "yes" else "no"
        html = None
        if predicted == "yes":
            tail.append(P.pcrc_extraction_turn(self.sample_image, self.sample_html))
            html = P.parse_html(self.client.complete(P.PCRC_SYSTEM, head + tail, config.MAX_NEW_TOKENS_EXTRACT, "extract"))
        recs = self.client.log[first:]
        return TableOutcome(dataset=self.dataset, method=self.method, doc_id=doc.doc_id, condition_id=cond.condition_id,
                           category=cond.category, table_id=table.table_id, gold=cond.label(table.table_id),
                           predicted=predicted, html_pred=html, rounds=len(answers), answers=answers,
                           prefill_actual=sum(r["computed_tokens"] for r in recs),
                           prefill_nocache=sum(r["prompt_tokens"] for r in recs),
                           decode=sum(r["decode_tokens"] for r in recs))

    def run_condition(self, doc: Document, index: RelevanceIndex, cond: Condition) -> List[TableOutcome]:
        text_of = {p.key: p.text for p in doc.paragraphs}
        head = P.pcrc_head(cond.text, [text_of[k] for k in index.shared_keys if k in text_of])
        return [self.run_table(head, t, [text_of[k] for k in index.exclusive_keys.get(t.key, []) if k in text_of], doc, cond)
                for t in doc.tables]

    def run_document(self, doc: Document, index: RelevanceIndex) -> Iterable[TableOutcome]:
        for cond in doc.conditions:
            yield from self.run_condition(doc, index, cond)
