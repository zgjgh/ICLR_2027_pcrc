"""Global configuration: paths, dataset aliases, and the default hyper-parameters
reported in Appendix A.3.1.

Paths are resolved from environment variables so that the same code runs on
any machine:

    PCRC_REPO_ROOT     repository root (default: the directory containing this package)
    PCRC_CORPUS_ROOT   directory holding one sub-directory per dataset
                       (default: <repo>/corpus)
    PCRC_OUTPUT_ROOT   where inference records and scores are written
                       (default: <repo>/outputs)
    QWEN_MODEL_PATH    Qwen2.5-VL-7B-Instruct checkpoint (HF id or local path)
    GLM_MODEL_PATH     GLM-4.5V checkpoint (Appendix A.3.7)
    NVEMBED_MODEL_PATH NV-Embed checkpoint used as the relevance encoder phi
    VLLM_API_BASE      OpenAI-compatible endpoint of a vLLM server (Appendix A.3.5)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(os.environ.get("PCRC_REPO_ROOT", Path(__file__).resolve().parent.parent))
CORPUS_ROOT = Path(os.environ.get("PCRC_CORPUS_ROOT", REPO_ROOT / "corpus"))
OUTPUT_ROOT = Path(os.environ.get("PCRC_OUTPUT_ROOT", REPO_ROOT / "outputs"))
DATA_DIR = REPO_ROOT / "data"

# Paper name -> bundled corpus directory / demo condition file stem.
DATASETS = {
    "fincontab": "fintabnet",
    "pubcontab": "pubtables",
    "degcontab": "wild",
}
DATASET_ALIASES = {**DATASETS, **{v: v for v in DATASETS.values()},
                   "fin": "fintabnet", "pub": "pubtables", "deg": "wild"}


def corpus_dir(dataset: str) -> Path:
    return CORPUS_ROOT / DATASET_ALIASES[dataset.lower()]


def per_doc_conditions_file(dataset: str) -> Path:
    """``data/demo_per_doc_<dataset>.json``: documents -> conditions and gold tables."""
    return DATA_DIR / f"demo_per_doc_{DATASET_ALIASES[dataset.lower()]}.json"


def demo_subset_ids_file(dataset: str) -> Path:
    """``data/demo_conditions_<dataset>.json``: condition ids of the demo subset."""
    return DATA_DIR / f"demo_conditions_{DATASET_ALIASES[dataset.lower()]}.json"


# ---------------------------------------------------------------- models
QWEN_MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct")
GLM_MODEL_PATH = os.environ.get("GLM_MODEL_PATH", "zai-org/GLM-4.5V")
NVEMBED_MODEL_PATH = os.environ.get("NVEMBED_MODEL_PATH", "nvidia/NV-Embed-v2")
VLLM_API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")

DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "sdpa"
# Qwen2.5-VL image budget: bounds the number of vision patch tokens per crop.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
MAX_NEW_TOKENS_JUDGE = 8          # yes / no / uncertain
MAX_NEW_TOKENS_EXTRACT = 4096     # one HTML table


# ------------------------------------------------- selection hyper-parameters
@dataclass(frozen=True)
class SelectionConfig:
    """Section 3.1 / Appendix A.3.1 defaults (Table 10 sweeps these four values)."""
    k_shared: int = 4          # K_shared: budget of shared related paragraphs S(D)
    tau_shared: float = 0.12   # tau_shared: threshold on g(P_j) = mean_i r(P_j, T_i)
    k_excl: int = 3            # K_excl: budget of exclusive related paragraphs E(T_i)
    tau_excl: float = 0.3      # tau_excl: threshold on r(P_j, T_i)
    lam: float = 1.0           # decay rate of the spatial factor exp(-lam * d_hat) (Table 12)
    relevance: str = "product" # product | semantic_only | spatial_only  (Table 9)


DEFAULT_SELECTION = SelectionConfig()

# Table 10 / Table 12 sweep grids
SWEEP_GRID = {
    "k_shared": [2, 4, 6, 8],
    "k_excl": [1, 2, 3, 4],
    "tau_shared": [0.06, 0.12, 0.18, 0.24],
    "tau_excl": [0.15, 0.3, 0.45, 0.6],
    "lam": [0.5, 1.0, 2.0, 3.0, 4.0],
}

# Condition strata (Section 4, Table 6): five categories, 20% each.
CONDITION_CATEGORIES = {
    "C1": "Lookup & Filter",
    "C2": "Numerical Reasoning",
    "C3": "Multi-Constraint Conjunction",
    "C4": "Multi-Step Reasoning",
    "C5": "Structural Conditions",
}
CATEGORY_TARGET_SHARE = {c: 0.20 for c in CONDITION_CATEGORIES}
