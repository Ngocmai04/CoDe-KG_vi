# llm_runtime.py
from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextGenerationPipeline


def set_seed(seed: int = 4000) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_chat_messages(system_text: str, user_text: str) -> List[dict]:
    """OpenAI / GPT-4o-mini compatible message list."""
    return [
        {"role": "system", "content": system_text.strip()},
        {"role": "user", "content": user_text.strip()},
    ]


def join_prompt_blocks(system_text: str, user_text: str) -> str:
    """Legacy single-string prompt used by local HF pipelines."""
    return system_text.strip() + "\n\n" + user_text.strip()


class LLM:
    """Local Hugging Face causal LM with 4-bit quantization."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipe = None

    def load(self) -> None:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.pipe = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer)

    def generate_batch(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        batch_size: int,
        temperature: float,
        top_k: int = 40,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        return_full_text: bool = False,
    ):
        if not prompts:
            return []

        if self.pipe is None:
            self.load()

        cur_batch = batch_size
        cur_tokens = max_new_tokens

        while True:
            try:
                outs = self.pipe(
                    list(prompts),
                    max_new_tokens=cur_tokens,
                    batch_size=cur_batch,
                    truncation=True,
                    return_full_text=return_full_text,
                    pad_token_id=self.tokenizer.eos_token_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    use_cache=True,
                )
                return outs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                time.sleep(0.2)

                if cur_batch > 1:
                    cur_batch = max(1, cur_batch // 2)
                    continue

                if cur_tokens > 256:
                    cur_tokens = max(256, cur_tokens // 2)
                    continue

                raise


class OpenAILLM:
    """
    Cloud backend for GPT-4o-mini (and compatible chat models).
    Expects OPENAI_API_KEY in the environment.
    Prompts may be plain strings (from build_prompt) or message dict lists.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "OpenAI backend requires `pip install openai` and OPENAI_API_KEY."
                ) from exc
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY is not set.")
            self._client = OpenAI(api_key=api_key)
        return self._client

    @staticmethod
    def _to_messages(prompt: Union[str, List[dict]]) -> List[dict]:
        if isinstance(prompt, list):
            return prompt
        # Single block: treat entire string as user turn (system embedded upstream)
        return [{"role": "user", "content": str(prompt).strip()}]

    def generate_batch(
        self,
        prompts: Sequence[Union[str, List[dict]]],
        max_new_tokens: int,
        batch_size: int,
        temperature: float,
        top_k: int = 40,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        return_full_text: bool = False,
    ):
        del batch_size, top_k, top_p, do_sample, repetition_penalty, return_full_text

        if not prompts:
            return []

        client = self._get_client()
        results = []

        for prompt in prompts:
            messages = self._to_messages(prompt)
            last_error: Optional[Exception] = None
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    text = (resp.choices[0].message.content or "").strip()
                    results.append([{"generated_text": text}])
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    time.sleep(0.5 * (attempt + 1))
            if last_error is not None:
                raise RuntimeError(
                    f"OpenAI generation failed for model={self.model_name}: {last_error}"
                ) from last_error

        return results


def create_llm(backend: str, model_name: str):
    """Factory: local HF vs OpenAI chat API."""
    backend = (backend or "local").strip().lower()
    if backend == "openai":
        return OpenAILLM(model_name=model_name)
    if backend == "local":
        return LLM(model_name=model_name)
    raise ValueError(f"Unknown llm backend '{backend}'. Use 'local' or 'openai'.")
