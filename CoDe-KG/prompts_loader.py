# prompts_loader.py
"""Select English or Vietnamese prompt modules."""
from __future__ import annotations

from types import ModuleType

from lang_config import LangCode, normalize_lang


def get_prompt_module(lang: str) -> ModuleType:
    code = normalize_lang(lang)
    if code == "vi":
        import prompts_vi as mod
    else:
        import prompts as mod
    return mod
