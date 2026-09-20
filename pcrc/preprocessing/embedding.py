"""Semantic similarity encoder phi (Section 3.1, "Semantic Similarity").

phi is NV-Embed, NVIDIA's pretrained generalist text embedding model, applied
to the OCR text of paragraphs and tables without fine-tuning:

    p_j = phi(OCR(P_j)),  t_i = phi(OCR(T_i)),
    s_sem(P_j, T_i) = max{0, cos(p_j, t_i)}.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .. import config


class NVEmbedEncoder:
    """Thin wrapper around the NV-Embed checkpoint (sentence-transformers / HF)."""

    def __init__(self, model_path: str = config.NVEMBED_MODEL_PATH, device: str = "cuda:0",
                 max_length: int = 4096, batch_size: int = 16):
        from transformers import AutoModel  # lazy import: scoring-only users need no torch

        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalised embeddings, one row per text (empty text -> zero vector)."""
        import torch

        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        idx = [i for i, t in enumerate(texts) if t and t.strip()]
        with torch.no_grad():
            for s in range(0, len(idx), self.batch_size):
                chunk = idx[s:s + self.batch_size]
                emb = self.model.encode([texts[i] for i in chunk], instruction="", max_length=self.max_length)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1).float().cpu().numpy()
                out[chunk] = emb
        return out


def semantic_similarity(par_emb: np.ndarray, tab_emb: np.ndarray) -> np.ndarray:
    """s_sem[j, i] = max(0, cos(p_j, t_i)) for L2-normalised embeddings."""
    return np.clip(par_emb @ tab_emb.T, 0.0, None)


class CachedEncoder:
    """Optional on-disk cache of embeddings keyed by (doc_id, block key)."""

    def __init__(self, encoder: NVEmbedEncoder, cache_file=None):
        import json
        self.encoder = encoder
        self.cache_file = cache_file
        self._cache = {}
        if cache_file is not None and cache_file.exists():
            self._cache = {k: np.asarray(v, dtype=np.float32) for k, v in json.loads(cache_file.read_text()).items()}

    def encode(self, keys: List[str], texts: List[str]) -> np.ndarray:
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            emb = self.encoder.encode([texts[i] for i in missing])
            for row, i in enumerate(missing):
                self._cache[keys[i]] = emb[row]
        return np.stack([self._cache[k] for k in keys]) if keys else np.zeros((0, 1), dtype=np.float32)

    def flush(self):
        import json
        if self.cache_file is not None:
            self.cache_file.write_text(json.dumps({k: v.tolist() for k, v in self._cache.items()}))
