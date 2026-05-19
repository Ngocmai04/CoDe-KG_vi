# lang_config.py
"""Language-aware defaults for the CoDe-KG pipeline."""
from __future__ import annotations

from typing import Literal

LangCode = Literal["en", "vi"]

SUPPORTED_LANGS: tuple[LangCode, ...] = ("en", "vi")

# CSV column used when --text_col is left at the sentinel default "auto"
TEXT_COLUMN_BY_LANG: dict[LangCode, str] = {
    "en": "context",
    "vi": "context_vi",
}

# Legacy single-column name still accepted for English-only CSVs
TEXT_COLUMN_FALLBACKS: dict[LangCode, tuple[str, ...]] = {
    "en": ("context", "abstract", "text"),
    "vi": ("context_vi", "abstract_vi", "text_vi"),
}

DEFAULT_BERT_MODEL: dict[LangCode, str] = {
    "en": "bert-large-uncased",
    "vi": "bert-base-multilingual-cased",
}

DEFAULT_BERT_OUT_DIR: dict[LangCode, str] = {
    "en": "models/bert_sentence_classifier",
    "vi": "models/bert_sentence_classifier_vi",
}

# Sentence-transformer used when lang=vi (multilingual semantic stages)
DEFAULT_EMBEDDING_MODEL: dict[LangCode, str | None] = {
    "en": None,
    "vi": "intfloat/multilingual-e5-base",
}

COLAB_LIGHT_BERT: dict[LangCode, str] = {
    "en": "bert-base-uncased",
    "vi": "bert-base-multilingual-cased",
}


def normalize_lang(lang: str) -> LangCode:
    code = (lang or "en").strip().lower()
    if code not in SUPPORTED_LANGS:
        raise ValueError(f"Unsupported --lang '{lang}'. Choose from: {', '.join(SUPPORTED_LANGS)}")
    return code  # type: ignore[return-value]


def resolve_text_column(lang: LangCode, text_col: str | None) -> str:
    """Map 'auto' / None to the language-specific context column."""
    if text_col and text_col != "auto":
        return text_col
    return TEXT_COLUMN_BY_LANG[lang]


def pick_available_text_column(df_columns: list[str], lang: LangCode, preferred: str) -> str:
    """Return preferred column if present, else first fallback for the language."""
    if preferred in df_columns:
        return preferred
    for col in TEXT_COLUMN_FALLBACKS[lang]:
        if col in df_columns:
            return col
    raise ValueError(
        f"No text column for lang='{lang}'. Expected one of "
        f"{(preferred, *TEXT_COLUMN_FALLBACKS[lang])} in columns: {df_columns}"
    )
