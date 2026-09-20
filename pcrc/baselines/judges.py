"""Downstream judges of the extract-then-judge pipelines (Section 5.1).

* ``QwenHTMLJudge``  Qwen run as a VLM judge that takes the condition, the
                     table image, and the upstream HTML and returns yes/no.
* ``TAPASJudge``     a non-VLM judge that operates on the structured table
                     derived from the upstream HTML (google/tapas-base-finetuned-wtq).
"""
from __future__ import annotations

from typing import List

from ..inference.engine import VLMEngine
from ..inference.prompts import parse_judgment

JUDGE_SYSTEM = "You judge whether a table satisfies a natural-language condition."


class QwenHTMLJudge:
    def __init__(self, engine: VLMEngine):
        self.engine = engine

    def judge(self, condition: str, table_image, html: str) -> str:
        user = [
            {"type": "text", "text": "[Table Image]"},
            {"type": "image", "image": table_image},
            {"type": "text", "text": f"\n\n[Extracted HTML]\n{html.strip()}\n\n[Condition]\n{condition.strip()}\n\n"
                                     "Does the table satisfy the condition? Return exactly yes or no."},
        ]
        raw = self.engine.generate_once([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
                                        max_new_tokens=4, label="judge")
        return parse_judgment(raw, allow_uncertain=False)


class TAPASJudge:
    """Structured-table judge: the condition is posed as a question over the
    HTML-derived table; a non-empty, non-negative answer counts as yes."""

    def __init__(self, model_name: str = "google/tapas-base-finetuned-wtq", device: str = "cuda:0"):
        from transformers import TapasForQuestionAnswering, TapasTokenizer

        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model = TapasForQuestionAnswering.from_pretrained(model_name).to(device).eval()
        self.device = device

    @staticmethod
    def html_to_frame(html: str):
        import pandas as pd
        from io import StringIO

        try:
            frame = pd.read_html(StringIO(html))[0]
        except Exception:
            return pd.DataFrame({"cell": [""]})
        frame.columns = [str(c) for c in frame.columns]
        return frame.fillna("").astype(str)

    def judge(self, condition: str, html: str) -> str:
        import torch

        table = self.html_to_frame(html)
        inputs = self.tokenizer(table=table, queries=[condition], padding="max_length", truncation=True,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        coords, agg = self.tokenizer.convert_logits_to_predictions(
            {k: v.cpu() for k, v in inputs.items()}, out.logits.cpu(), out.logits_aggregation.cpu())
        answered = bool(coords and coords[0]) or (agg and agg[0] != 0)
        return "yes" if answered else "no"
