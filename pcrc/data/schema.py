"""Common example schema of the three datasets (Appendix A.1.1, Table 4).

A document is an ordered sequence of page images. Every page carries one or
more tables and surrounding paragraph text. Each table has an identifier, a
crop, OCR text, and its reference HTML. Each document is paired with
natural-language conditions, and for every condition every table receives
a binary yes/no label. A system returns HTML only for the tables it judges
to satisfy the condition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

TableId = Union[int, str]


@dataclass
class Paragraph:
    """A text block of the parsed document (candidate context P_j)."""
    key: str                      # globally unique block key, e.g. "ADS_2007_p104__paragraph_1"
    page: int
    bbox: List[float]             # [x0, y0, x1, y1] in page coordinates
    text: str                     # OCR / text-layer content (masked for DegConTab, see parsing.py)
    block_type: str = "paragraph" # paragraph | caption | title | ...


@dataclass
class Table:
    """A candidate table T_i produced by the front-end parser."""
    key: str
    table_id: TableId
    page: int
    bbox: List[float]
    image_path: str
    ocr_text: str                 # unstructured character sequence, never a structured table
    html_ref: str = ""            # reference HTML (used for scoring and condition generation only)


@dataclass
class Condition:
    condition_id: str
    text: str
    category: str = ""            # C1..C5 (Table 6)
    gold_yes_tables: List[TableId] = field(default_factory=list)

    def label(self, table_id: TableId) -> str:
        return "yes" if str(table_id) in {str(t) for t in self.gold_yes_tables} else "no"


@dataclass
class Document:
    doc_id: str
    n_pages: int
    page_sizes: Dict[int, List[float]]        # page -> [width, height]
    paragraphs: List[Paragraph]
    tables: List[Table]
    conditions: List[Condition] = field(default_factory=list)

    def paragraph_by_key(self, key: str) -> Optional[Paragraph]:
        for p in self.paragraphs:
            if p.key == key:
                return p
        return None


@dataclass
class RelevanceIndex:
    """Output of PCRC preprocessing: I(D) = (S(D), {E(T_i)}) (Section 3.1)."""
    doc_id: str
    shared_keys: List[str]                          # S(D), in reading order
    exclusive_keys: Dict[str, List[str]]            # table key -> E(T_i), by descending r
    scores: Dict[str, Dict[str, float]] = field(default_factory=dict)  # table key -> {paragraph key -> r}
    doc_scores: Dict[str, float] = field(default_factory=dict)         # paragraph key -> g(P_j)


@dataclass
class TableOutcome:
    """One (document, condition, table) decision produced by an inference method."""
    dataset: str
    method: str
    doc_id: str
    condition_id: str
    category: str
    table_id: TableId
    gold: str                     # yes | no
    predicted: str                # yes | no
    html_pred: Optional[str]      # extracted HTML when predicted == yes
    rounds: int                   # judgment calls issued for this table
    answers: List[str]            # raw judge answers per round
    prefill_actual: int           # prefill tokens actually encoded (cache misses)
    prefill_nocache: int          # prefill tokens the same calls cost without cache reuse
    decode: int                   # generated tokens

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d["table_id"] = str(self.table_id)
        return d
