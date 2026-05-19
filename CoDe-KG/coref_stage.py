# coref_stage.py
import re
from typing import Dict, List, Any, Optional

from tokenize_paper import parse_tokenized_pairs
from prompts_loader import get_prompt_module
from embedding_runtime import TextEmbedder


def _extract_coref_blocks(raw: str) -> List[Dict[str, Any]]:
    """
    Regex-first extraction for blocks that contain:
    Expression, StartToken, EndToken, RefersTo
    Works even if JSON is slightly malformed.
    """
    blocks = []

    candidates = re.findall(r'\{[^{}]*"Expression"[^{}]*\}', raw, flags=re.DOTALL)
    if not candidates:
        candidates = re.findall(r'\{[^{}]*Expression[^{}]*\}', raw, flags=re.DOTALL)

    for c in candidates:
        exp = re.search(r'Expression"\s*:\s*"([^"]+)"', c)
        if exp is None:
            exp = re.search(r'Expression\s*:\s*"([^"]+)"', c)

        st = re.search(r'StartToken"\s*:\s*(\d+)', c)
        if st is None:
            st = re.search(r'StartToken\s*:\s*(\d+)', c)

        en = re.search(r'EndToken"\s*:\s*(\d+)', c)
        if en is None:
            en = re.search(r'EndToken\s*:\s*(\d+)', c)

        rf = re.search(r'RefersTo"\s*:\s*"([^"]+)"', c)
        if rf is None:
            rf = re.search(r'RefersTo\s*:\s*"([^"]+)"', c)

        if exp and st and en and rf:
            blocks.append({
                "Expression": exp.group(1),
                "StartToken": int(st.group(1)),
                "EndToken": int(en.group(1)),
                "RefersTo": rf.group(1),
            })

    return blocks


def build_coref_prompt(tokenized_text: str, lang: str = "en") -> str:
    prompts = get_prompt_module(lang)
    return prompts.build_prompt(
        prompts.COREf_FICL_SYSTEM,
        prompts.COREf_FICL_USER,
        tokenized_text=tokenized_text,
    )


def apply_coref_switch(tokenized_text: str, corefs: List[Dict[str, Any]]) -> str:
    tokens = parse_tokenized_pairs(tokenized_text)
    token_map = {idx: tok for tok, idx in tokens}

    corefs_sorted = sorted(corefs, key=lambda x: (x["StartToken"], x["EndToken"]))
    replacements: Dict[int, str | None] = {}

    for c in corefs_sorted:
        start_token = int(c["StartToken"])
        end_token = int(c["EndToken"])
        refers_to = str(c["RefersTo"]).strip()

        for token_idx in range(start_token, end_token + 1):
            if token_idx == start_token:
                replacements[token_idx] = refers_to
            else:
                replacements[token_idx] = None

    result_tokens: List[str] = []
    for tok, idx in tokens:
        if idx in replacements:
            if replacements[idx] is not None:
                result_tokens.append(replacements[idx])
        else:
            result_tokens.append(tok)

    text = " ".join(result_tokens)

    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_coref_output(raw: str) -> List[Dict[str, Any]]:
    return _extract_coref_blocks(raw)


def score_coref_referent(
    expression: str,
    refers_to: str,
    embedder: Optional[TextEmbedder] = None,
) -> Optional[float]:
    """Optional semantic score for a coref link (multilingual embedder when lang=vi)."""
    if embedder is None or not embedder.enabled:
        return None
    vecs = embedder.encode([expression, refers_to])
    if vecs.shape[0] < 2:
        return None
    return float((vecs[0] * vecs[1]).sum())
