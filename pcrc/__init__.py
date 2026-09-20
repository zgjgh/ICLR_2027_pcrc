"""PCRC: Progressive Context-Reusable Cache for query-conditioned table extraction.

Package layout mirrors the paper:

    pcrc.preprocessing   Section 3.1  table-centered relevance index
    pcrc.inference       Section 3.2  cache-frame inference (Algorithm 1), prompts (Appendix A.2)
    pcrc.baselines       Section 5.1  single-shot and modular pipelines
    pcrc.evaluation      Section 3.3  judgment / extraction / composite quality, token cost
    pcrc.datasets        Section 4, Appendix A.1  condition generation and DegConTab synthesis
"""

__version__ = "1.0.0"
