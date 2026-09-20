"""Prompt templates (Appendix A.2), reproduced block for block.

Two templates are used:

* ``SINGLE_SHOT``  (A.2.1) the Qwen baseline: one decoding pass that decides
  whether the table satisfies the condition and, if so, extracts it,
  returning either ``no`` or the table HTML.
* ``PCRC``        (A.2.2) the cache-frame contract. The blocks up to the
  shared related paragraphs form the fixed head; the table image, table OCR,
  and exclusive related paragraphs form the table-specific tail, and the
  extraction turn (sample image, sample HTML, request) is appended only
  after a positive judgment.

Each block is materialised as a *segment*: an ordered list of chat-content
items (``{"type": "text", ...}`` / ``{"type": "image", ...}``) so that the
engine can encode it incrementally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# ----------------------------------------------------------- A.2.1 single-shot
SINGLE_SHOT_SYSTEM = "You are performing conditioned table extraction."

SINGLE_SHOT_HEADER = """[Input]
You are given:
- a table image;
- OCR text from this table image. The OCR text contains table characters only and is not a structured table;
- a natural-language condition;
- one sample table image;
- the reference HTML for the sample table.

[Goal]
First decide whether the table satisfies the condition. If it does, extract the table as HTML by following the correspondence between the sample table image and its reference HTML. If it does not, return no.

[Instructions]
1. Use the table image to recover layout, row/column structure, headers, and cell spans.
2. Use the OCR text only as unstructured textual evidence for the table content.
3. Use the sample table image and sample HTML only as a formatting demonstration; do not copy their content.
4. If the candidate table satisfies the condition, return only the extracted HTML table.
5. If the candidate table does not satisfy the condition, return exactly no."""

# ------------------------------------------------------------------ A.2.2 PCRC
PCRC_SYSTEM = "You are a context-aware conditioned table extractor."

PCRC_HEADER = """[Input]
You are given:
- a natural-language condition;
- paragraphs shared across the tables of this document;
- a table image;
- table OCR, containing characters only and no structure;
- paragraphs specific to this table;
- once the table is accepted, a sample table image and its reference HTML.

[Goal]
Use the image, OCR, and related paragraphs to decide whether the table satisfies the condition: answer yes if it clearly does, no if it clearly does not, and uncertain if the evidence is insufficient. An accepted table is then extracted as HTML following the sample image/HTML correspondence.

[Instructions]
1. Use the table image to recover visual layout, row/column structure, headers, and cell spans.
2. Use the OCR text as unstructured textual evidence for table content.
3. Use related paragraphs for table topic, variable meanings, measurement context, and header scope.
4. For the judgment, return exactly yes, no, or uncertain.
5. For extraction, use the sample table image and sample HTML only as a formatting demonstration; do not copy their content. Return only the extracted HTML table."""

EXTRACTION_REQUEST = "[Extraction Request]\nExtract the table as HTML following the sample image/HTML correspondence."
FORCED_DECISION_REQUEST = "[Final Judgment]\nNo further paragraphs are available. Decide now: return exactly yes or no."
JUDGE_ANSWERS = ("yes", "no", "uncertain")


@dataclass
class Segment:
    """One prompt block, encoded as a unit by the engine.

    ``parts`` is an ordered list of chat-content items, so a heading such as
    ``[Table Image]`` precedes the image it introduces exactly as in A.2.2.
    """
    name: str
    parts: List[dict] = field(default_factory=list)

    @classmethod
    def text(cls, name: str, text: str) -> "Segment":
        return cls(name, [{"type": "text", "text": text}])

    def content(self) -> List[dict]:
        return list(self.parts)


def _numbered(paragraphs: List[str], prefix: str) -> str:
    return "\n".join(f"{prefix}{i + 1}. {p.strip()}" for i, p in enumerate(paragraphs)) if paragraphs else "(none)"


# ----------------------------------------------------------- PCRC segments
def pcrc_head(condition: str, shared_paragraphs: List[str]) -> List[Segment]:
    """Fixed head Pi_head = [iota_sys; iota_judge; q; S(D)] (Section 3.2, Eq. 4)."""
    return [
        Segment.text("header", PCRC_HEADER),
        Segment.text("condition", f"\n\n[Condition]\n{condition.strip()}"),
        Segment.text("shared", f"\n\n[Shared Related Paragraphs]\n{_numbered(shared_paragraphs, 'S')}"),
    ]


def pcrc_tail(table_image, table_ocr: str) -> Segment:
    """Table-specific tail: cropped image + OCR (Algorithm 1, line 2)."""
    return Segment("tail", [
        {"type": "text", "text": "\n\n[Table Image]\n"},
        {"type": "image", "image": table_image},
        {"type": "text", "text": f"\n\n[Table OCR]\n{table_ocr.strip()}"},
    ])


def pcrc_exclusive(paragraph: str, k: int) -> Segment:
    """One exclusive related paragraph e_k^(i), appended progressively."""
    header = "\n\n[Exclusive Related Paragraphs]\n" if k == 1 else "\n"
    return Segment.text(f"excl_{k}", f"{header}E{k}. {paragraph.strip()}")


def pcrc_judge_request(forced: bool = False) -> Segment:
    return Segment.text("forced" if forced else "judge",
                        f"\n\n{FORCED_DECISION_REQUEST}" if forced else "\n\n[Judgment]\nyes, no, or uncertain:")


def pcrc_extraction_turn(sample_image, sample_html: str) -> Segment:
    """Extraction turn appended after a positive judgment (Algorithm 1, line 7)."""
    return Segment("extract", [
        {"type": "text", "text": "\n\n[Sample Table Image]\n"},
        {"type": "image", "image": sample_image},
        {"type": "text", "text": f"\n\n[Sample HTML]\n{sample_html.strip()}\n\n{EXTRACTION_REQUEST}"},
    ])


# ---------------------------------------------------- single-shot messages
def single_shot_messages(condition: str, table_image, table_ocr: str, sample_image, sample_html: str) -> List[dict]:
    user = [
        {"type": "text", "text": SINGLE_SHOT_HEADER},
        {"type": "text", "text": "\n\n[Table Image]"},
        {"type": "image", "image": table_image},
        {"type": "text", "text": f"\n\n[Table OCR]\n{table_ocr.strip()}\n\n[Condition]\n{condition.strip()}\n\n[Sample Table Image]"},
        {"type": "image", "image": sample_image},
        {"type": "text", "text": f"\n\n[Sample HTML]\n{sample_html.strip()}"},
    ]
    return [{"role": "system", "content": SINGLE_SHOT_SYSTEM}, {"role": "user", "content": user}]


# --------------------------------------------------------- answer parsing
def parse_judgment(text: str, allow_uncertain: bool = True) -> str:
    """Map a raw judge answer to yes / no / uncertain (unknown -> uncertain)."""
    t = text.strip().lower().strip(".:'\" ")
    for a in JUDGE_ANSWERS:
        if t.startswith(a):
            return a if (allow_uncertain or a != "uncertain") else "no"
    return "uncertain" if allow_uncertain else "no"


def parse_single_shot(text: str) -> Optional[str]:
    """Return the extracted HTML, or None when the answer is ``no``."""
    t = text.strip()
    if t.lower().startswith("no") and "<table" not in t.lower():
        return None
    start, end = t.lower().find("<table"), t.lower().rfind("</table>")
    return t[start:end + len("</table>")] if start >= 0 and end > start else (t or None)


def parse_html(text: str) -> str:
    t = text.strip()
    start, end = t.lower().find("<table"), t.lower().rfind("</table>")
    return t[start:end + len("</table>")] if start >= 0 and end > start else t
