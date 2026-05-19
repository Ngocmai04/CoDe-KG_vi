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
