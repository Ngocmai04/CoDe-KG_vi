# sentence_split_stage.py
from __future__ import annotations

import re
from typing import List, Optional

import nltk
from nltk.tokenize import sent_tokenize

from embedding_runtime import TextEmbedder


def setup_nltk() -> None:
    for resource in ("punkt", "punkt_tab"):
        try:
            sent_tokenize("Test sentence.", language="english")
            return
        except LookupError:
            nltk.download(resource, quiet=True)


def _split_vietnamese_regex(text: str) -> List[str]:
    """Fallback: split on sentence-ending punctuation (Vietnamese/ASCII)."""
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_vietnamese_underthesea(text: str) -> List[str]:
    from underthesea import sent_tokenize as vi_sent_tokenize

    return [s.strip() for s in vi_sent_tokenize(text.strip()) if s.strip()]


def split_sentences_vi(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    try:
        return _split_vietnamese_underthesea(text)
    except ImportError:
        return _split_vietnamese_regex(text)


def _cosine(a, b) -> float:
    return float((a * b).sum())


def _merge_over_split_sentences(
    sentences: List[str],
    embedder: TextEmbedder,
    similarity_threshold: float = 0.88,
) -> List[str]:
    """Merge consecutive fragments that are likely one sentence (common with vi tokenizers)."""
    if len(sentences) < 2 or not embedder.enabled:
        return sentences

    embs = embedder.encode(sentences)
    merged: List[str] = [sentences[0]]
    merged_embs = [embs[0]]

    for i in range(1, len(sentences)):
        sim = _cosine(merged_embs[-1], embs[i])
        # Merge short tail fragments or very high similarity pairs
        prev_short = len(merged[-1]) < 40
        cur_short = len(sentences[i]) < 40
        if sim >= similarity_threshold and (prev_short or cur_short):
            merged[-1] = f"{merged[-1]} {sentences[i]}".strip()
            merged_embs[-1] = (merged_embs[-1] + embs[i]) / 2.0
            merged_embs[-1] /= max(float((merged_embs[-1] ** 2).sum()) ** 0.5, 1e-8)
        else:
            merged.append(sentences[i])
            merged_embs.append(embs[i])

    return merged


def split_sentences(
    text: str,
    lang: str = "en",
    embedder: Optional[TextEmbedder] = None,
) -> List[str]:
    if not text or not text.strip():
        return []

    lang_code = (lang or "en").strip().lower()

    if lang_code == "vi":
        sents = split_sentences_vi(text)
        if embedder is not None and embedder.enabled:
            sents = _merge_over_split_sentences(sents, embedder)
        return sents

    setup_nltk()
    try:
        sents = [s.strip() for s in sent_tokenize(text.strip(), language="english") if s.strip()]
    except LookupError:
        setup_nltk()
        sents = [s.strip() for s in sent_tokenize(text.strip()) if s.strip()]
    return sents
