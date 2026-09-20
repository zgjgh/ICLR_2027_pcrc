"""Frontier VLMs served through API endpoints (Appendix A.3.7 and A.3.8).

Claude Opus 4.5, GPT-5.2, and Qwen3-VL-Plus do not expose internal kv
states, so PCRC's cache reuse cannot be implemented on them and no
token-level cache comparison is possible. They are evaluated as single-shot
baselines under the same input/output protocol as the Qwen2.5-VL-7B
baseline (Appendix A.2.1), and in the extraction-only context ablation
(Condition A: image + OCR; Condition B: + the shared paragraph and the
top-2 exclusive paragraphs; same one-shot HTML example).

Set ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` / ``DASHSCOPE_API_KEY``.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Dict, List, Optional

from . import prompts as P

BACKBONES: Dict[str, dict] = {
    "claude-opus-4.5": {"provider": "anthropic", "model": "claude-opus-4-5"},
    "gpt-5.2": {"provider": "openai", "model": "gpt-5.2"},
    "qwen3-vl-plus": {"provider": "openai", "model": "qwen3-vl-plus",
                      "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "key_env": "DASHSCOPE_API_KEY"},
}


def _png_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class APIBackbone:
    def __init__(self, name: str):
        spec = BACKBONES[name]
        self.name, self.provider, self.model = name, spec["provider"], spec["model"]
        if self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
        else:
            from openai import OpenAI
            self.client = OpenAI(base_url=spec.get("base_url"), api_key=os.environ.get(spec.get("key_env", "OPENAI_API_KEY")))

    def complete(self, system: str, content: List[dict], max_tokens: int) -> str:
        """``content``: chat parts as produced by ``pcrc.inference.prompts`` (text / image)."""
        if self.provider == "anthropic":
            parts = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _png_b64(c["image"])}}
                     if c["type"] == "image" else {"type": "text", "text": c["text"]} for c in content]
            resp = self.client.messages.create(model=self.model, system=system, max_tokens=max_tokens, temperature=0,
                                               messages=[{"role": "user", "content": parts}])
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + _png_b64(c["image"])}}
                 if c["type"] == "image" else {"type": "text", "text": c["text"]} for c in content]
        resp = self.client.chat.completions.create(model=self.model, temperature=0, max_tokens=max_tokens,
                                                   messages=[{"role": "system", "content": system},
                                                             {"role": "user", "content": parts}])
        return resp.choices[0].message.content or ""

    # ------------------------------------------------------- single-shot
    def single_shot(self, condition: str, table_image, table_ocr: str, sample_image, sample_html: str) -> Optional[str]:
        msgs = P.single_shot_messages(condition, table_image, table_ocr, sample_image, sample_html)
        return P.parse_single_shot(self.complete(msgs[0]["content"], msgs[1]["content"], 4096))

    # ------------------------------------------- extraction-only ablation
    def extract(self, table_image, table_ocr: str, sample_image, sample_html: str,
                paragraphs: Optional[List[str]] = None) -> str:
        """Condition A (paragraphs=None) or Condition B (shared + top-2 exclusive)."""
        text = ("[Goal]\nExtract the table as HTML following the correspondence between the sample table image "
                "and its reference HTML. Return only the HTML table.\n\n[Table Image]")
        content = [{"type": "text", "text": text}, {"type": "image", "image": table_image},
                   {"type": "text", "text": f"\n\n[Table OCR]\n{table_ocr.strip()}"}]
        if paragraphs:
            content.append({"type": "text", "text": "\n\n[Related Paragraphs]\n" +
                            "\n".join(f"P{i + 1}. {p.strip()}" for i, p in enumerate(paragraphs))})
        content += [{"type": "text", "text": "\n\n[Sample Table Image]"}, {"type": "image", "image": sample_image},
                    {"type": "text", "text": f"\n\n[Sample HTML]\n{sample_html.strip()}\n\n{P.EXTRACTION_REQUEST}"}]
        return P.parse_html(self.complete("You are performing table extraction.", content, 4096))
