# paths.py
"""Resolve data files relative to the CoDe-KG package (works on Kaggle regardless of cwd)."""
from __future__ import annotations

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def package_path(*parts: str) -> str:
    return os.path.join(PACKAGE_DIR, *parts)


def resolve_existing_path(
    path: str,
    *,
    fallbacks: tuple[str, ...] = (),
    description: str = "file",
) -> str:
    """
    Resolve path to an existing file:
    1. As given (absolute or relative to cwd)
    2. Relative to CoDe-KG package directory
    3. Optional fallback filenames under the package dir
    """
    candidates: list[str] = []

    if os.path.isabs(path):
        candidates.append(path)
    else:
        candidates.append(os.path.abspath(path))
        candidates.append(package_path(path))

    for name in fallbacks:
        candidates.append(package_path(name))

    seen: set[str] = set()
    for c in candidates:
        c = os.path.normpath(c)
        if c in seen:
            continue
        seen.add(c)
        if os.path.isfile(c):
            return c

    tried = "\n  - ".join(sorted(seen))
    raise FileNotFoundError(
        f"{description} not found: '{path}'\n"
        f"Tried:\n  - {tried}\n"
        f"Place train.csv in {PACKAGE_DIR} or pass --bert_train_csv /full/path/train.csv"
    )


BERT_TEXT_COL_ALIASES: tuple[str, ...] = (
    "Sentence",
    "sentence",
    "text",
    "Text",
    "content",
    "abstract",
    "Abstract",
)
BERT_LABEL_COL_ALIASES: tuple[str, ...] = (
    "label",
    "Label",
    "labels",
    "Labels",
    "class",
    "category",
)


def resolve_csv_column(
    columns,
    preferred: str,
    aliases: tuple[str, ...],
    *,
    role: str = "column",
) -> str:
    """Resolve a CSV column name (supports 'auto' and case-insensitive match)."""
    cols = list(columns)
    col_set = set(cols)
    lower_to_orig = {c.lower(): c for c in cols}

    if preferred and preferred != "auto":
        if preferred in col_set:
            return preferred
        if preferred.lower() in lower_to_orig:
            return lower_to_orig[preferred.lower()]

    for name in aliases:
        if name in col_set:
            return name
        if name.lower() in lower_to_orig:
            return lower_to_orig[name.lower()]

    raise KeyError(
        f"Could not resolve {role} column (preferred={preferred!r}). "
        f"Available columns: {cols}. "
        f"Expected one of: {(preferred, *aliases) if preferred != 'auto' else aliases}"
    )
