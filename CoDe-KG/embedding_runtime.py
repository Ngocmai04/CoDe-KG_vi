# embedding_runtime.py
"""Lazy multilingual sentence embeddings (used when --lang vi)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from lang_config import LangCode, DEFAULT_EMBEDDING_MODEL, normalize_lang


class TextEmbedder:
    """Wraps sentence-transformers; no-op for English (model_name=None)."""

    def __init__(self, lang: str, model_name: Optional[str] = None):
        self.lang: LangCode = normalize_lang(lang)
        self.model_name = model_name if model_name is not None else DEFAULT_EMBEDDING_MODEL[self.lang]
        self._model = None

    @property
    def enabled(self) -> bool:
        return bool(self.model_name)

    def load(self) -> None:
        if not self.enabled or self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if not self.enabled:
            raise RuntimeError("TextEmbedder.encode called but no embedding model is configured for this language.")
        self.load()
        # E5 models expect "query: " / "passage: " prefixes for best results; use passage for sentences.
        prefix = "passage: " if self.model_name and "e5" in self.model_name.lower() else ""
        inputs = [prefix + t for t in texts]
        return np.asarray(self._model.encode(inputs, normalize_embeddings=True), dtype=np.float32)
