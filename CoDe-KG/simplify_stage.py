# simplify_stage.py
import re
from typing import List, Dict, Any

from prompts_loader import get_prompt_module


def extract_simplified_sentences(raw: str) -> List[str]:
    sents = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("S"):
            continue

        if "->" in line:
            parts = line.split("->", 1)
        elif "→" in line:
            parts = line.split("→", 1)
        else:
            continue

        if len(parts) == 2:
            sent = parts[1].strip()
            if sent:
                sents.append(sent)
    return sents


def _prompts_for_type(sentence_type: str, lang: str):
    p = get_prompt_module(lang)
    if sentence_type == "Compound Sentence":
        return p.SIMPLIFY_COMPOUND_SYSTEM, p.SIMPLIFY_COMPOUND_USER, 1024, 0.7
    if sentence_type == "Complex Sentence":
        return p.SIMPLIFY_COMPLEX_SYSTEM, p.SIMPLIFY_COMPLEX_USER, 512, 0.3
    if sentence_type == "Compound-Complex Sentence":
        return (
            p.SIMPLIFY_COMPOUND_COMPLEX_SYSTEM,
            p.SIMPLIFY_COMPOUND_COMPLEX_USER,
            1024,
            0.7,
        )
    raise ValueError(f"Unsupported sentence_type: {sentence_type}")


def simplify_sentences_for_class(
    llm,
    sentence_type: str,
    sentences: List[str],
    batch_size: int,
    lang: str = "en",
) -> List[Dict[str, Any]]:
    rows = []
    if not sentences:
        return rows

    sys, usr, max_new_tokens, temperature = _prompts_for_type(sentence_type, lang)
    prompts_mod = get_prompt_module(lang)
    prompts = [prompts_mod.build_prompt(sys, usr, sentence=s) for s in sentences]

    outs = llm.generate_batch(
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        temperature=temperature,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        return_full_text=False,
    )

    for orig, out in zip(sentences, outs):
        raw = out[0]["generated_text"].strip()
        simplified = extract_simplified_sentences(raw)
        rows.append({
            "sentence_type": sentence_type,
            "original_sentence": orig,
            "simplified_sentences": simplified,
            "raw_output": raw,
        })

    return rows
