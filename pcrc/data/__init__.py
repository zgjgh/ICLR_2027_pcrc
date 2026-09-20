"""Data access: schema (Appendix A.1.1) and corpus/condition readers."""
from .schema import Condition, Document, Paragraph, RelevanceIndex, Table, TableOutcome  # noqa: F401
from .corpus import iter_documents, load_conditions, load_document, load_image, load_index, save_index  # noqa: F401
