# run_pipeline.py
import argparse
import os
import pandas as pd

from tokenize_paper import tokens_with_indices, format_pairs
from llm_runtime import set_seed, create_llm
from coref_stage import build_coref_prompt, parse_coref_output, apply_coref_switch
from sentence_split_stage import split_sentences
from bert_sentence_classifier import BertSentenceClassifier, BertClassifierConfig
from simplify_stage import simplify_sentences_for_class
from relation_extract_stage import extract_relationships
from lang_config import (
    normalize_lang,
    resolve_text_column,
    pick_available_text_column,
    DEFAULT_BERT_MODEL,
    DEFAULT_BERT_OUT_DIR,
    COLAB_LIGHT_BERT,
)
from embedding_runtime import TextEmbedder
from paths import package_path, resolve_existing_path

MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DEFAULT_BERT_TRAIN_CSV = package_path("train.csv")
LLAMA_31_8B = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_33_70B = "meta-llama/Llama-3.3-70B-Instruct"
LIGHT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
OPENAI_DEFAULT = "gpt-4o-mini"


def read_abstracts(args) -> pd.DataFrame:
    lang = normalize_lang(args.lang)

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        preferred_col = resolve_text_column(lang, args.text_col)
        text_col = pick_available_text_column(list(df.columns), lang, preferred_col)

        df = df.reset_index(drop=True)
        if args.id_col and args.id_col in df.columns:
            df["doc_id"] = df[args.id_col].astype(str)
        elif "doc_id" in df.columns:
            df["doc_id"] = df["doc_id"].astype(str)
        else:
            df["doc_id"] = [f"doc_{i}" for i in range(len(df))]

        series = df[text_col]
        missing_mask = series.isna() | (series.astype(str).str.strip() == "")
        if missing_mask.any():
            bad_ids = df.loc[missing_mask, "doc_id"].astype(str).tolist()
            preview = ", ".join(bad_ids[:8])
            suffix = f" (+{len(bad_ids) - 8} more)" if len(bad_ids) > 8 else ""
            raise ValueError(
                f"Missing or empty text for lang='{lang}' in column '{text_col}'. "
                f"doc_id examples: {preview}{suffix}"
            )

        df["abstract"] = series.astype(str)
        print(f"[data] lang={lang} text_col={text_col} rows={len(df)}")
        return df[["doc_id", "abstract"]]

    if args.abstract:
        if not str(args.abstract).strip():
            raise ValueError(f"--abstract is empty for lang='{lang}'.")
        return pd.DataFrame([{"doc_id": "doc_0", "abstract": args.abstract}])

    text = input("Enter abstract: ").strip()
    if not text:
        raise ValueError(f"stdin text is empty for lang='{lang}'.")
    return pd.DataFrame([{"doc_id": "doc_0", "abstract": text}])


def main():
    parser = argparse.ArgumentParser(
        description="CoDe-KG pipeline with English / Vietnamese execution modes.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["en", "vi"],
        help="Processing language: en -> context column, vi -> context_vi column",
    )
    parser.add_argument("--input_csv", type=str, default=None, help="CSV containing documents")
    parser.add_argument(
        "--text_col",
        type=str,
        default="auto",
        help="Text column name, or 'auto' to pick context / context_vi from --lang",
    )
    parser.add_argument("--id_col", type=str, default=None, help="Optional doc id column")

    parser.add_argument("--abstract", type=str, default=None, help="Single document (no CSV)")
    parser.add_argument("--out_csv", type=str, required=True, help="Output CSV path")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=4000)

    parser.add_argument(
        "--llm-backend",
        type=str,
        default="local",
        choices=["local", "openai"],
        help="local = HuggingFace; openai = GPT-4o-mini (OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default=OPENAI_DEFAULT,
        help="Chat model when --llm-backend openai",
    )

    parser.add_argument("--coref-model", type=str, default=None, help="Model for coreference stage")
    parser.add_argument("--compound-model", type=str, default=None, help="Compound simplification model")
    parser.add_argument("--complex-model", type=str, default=None, help="Complex simplification model")
    parser.add_argument("--cc-model", type=str, default=None, help="Compound-complex simplification model")
    parser.add_argument("--relation-model", type=str, default=None, help="Relation extraction model")
    parser.add_argument(
        "--colab-light",
        action="store_true",
        help="Lightweight preset (small model + smaller batch + lighter BERT)",
    )

    parser.add_argument("--bert_out_dir", type=str, default=None)
    parser.add_argument("--bert_model_name", type=str, default=None)
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="Sentence-transformer for vi refinement (default: multilingual-e5-base when --lang vi)",
    )
    parser.add_argument("--train_bert_if_missing", action="store_true")
    parser.add_argument(
        "--bert_train_csv",
        type=str,
        default=DEFAULT_BERT_TRAIN_CSV,
        help=f"BERT fine-tune CSV (default: {DEFAULT_BERT_TRAIN_CSV})",
    )
    parser.add_argument("--bert_text_col", type=str, default="sentence")
    parser.add_argument("--bert_label_col", type=str, default="label")
    parser.add_argument("--bert_epochs", type=int, default=3)
    parser.add_argument("--bert_train_batch_size", type=int, default=8)
    parser.add_argument("--bert_lr", type=float, default=2e-5)

    args = parser.parse_args()
    lang = normalize_lang(args.lang)
    set_seed(args.seed)

    default_llm = args.openai_model if args.llm_backend == "openai" else MIXTRAL
    if args.coref_model is None:
        args.coref_model = default_llm
    if args.compound_model is None:
        args.compound_model = default_llm if args.llm_backend == "openai" else LLAMA_31_8B
    if args.complex_model is None:
        args.complex_model = default_llm if args.llm_backend == "openai" else LLAMA_33_70B
    if args.cc_model is None:
        args.cc_model = default_llm
    if args.relation_model is None:
        args.relation_model = default_llm

    if args.bert_model_name is None:
        args.bert_model_name = DEFAULT_BERT_MODEL[lang]
    if args.bert_out_dir is None:
        args.bert_out_dir = DEFAULT_BERT_OUT_DIR[lang]

    if args.colab_light:
        if args.llm_backend == "local":
            args.coref_model = LIGHT_MODEL
            args.compound_model = LIGHT_MODEL
            args.complex_model = LIGHT_MODEL
            args.cc_model = LIGHT_MODEL
            args.relation_model = LIGHT_MODEL
        args.batch_size = min(args.batch_size, 2)
        if args.bert_model_name in ("bert-large-uncased", "bert-base-uncased"):
            args.bert_model_name = COLAB_LIGHT_BERT[lang]

    df_abs = read_abstracts(args)

    llm_cache = {}

    def get_llm(model_name: str):
        if model_name not in llm_cache:
            llm_cache[model_name] = create_llm(args.llm_backend, model_name)
        return llm_cache[model_name]

    embedder = TextEmbedder(lang=lang, model_name=args.embedding_model)
    if embedder.enabled:
        print(f"[embeddings] lang={lang} model={embedder.model_name}")

    bert_cfg = BertClassifierConfig(model_name=args.bert_model_name, out_dir=args.bert_out_dir)
    bert = BertSentenceClassifier(bert_cfg)
    if bert.exists():
        bert.load()
    else:
        if args.train_bert_if_missing:
            bert_train_csv = resolve_existing_path(
                args.bert_train_csv,
                fallbacks=("train.csv",),
                description="BERT training CSV",
            )
            bert.train_if_missing(
                train_csv=bert_train_csv,
                text_col=args.bert_text_col,
                label_col=args.bert_label_col,
                epochs=args.bert_epochs,
                batch_size=args.bert_train_batch_size,
                lr=args.bert_lr,
            )
        else:
            raise ValueError(
                f"No finetuned BERT found at {args.bert_out_dir}. Use --train_bert_if_missing."
            )

    all_rows = []

    print("=== Pipeline Config ===")
    print(f"lang={lang}")
    print(f"llm_backend={args.llm_backend}")
    print(f"coref_model={args.coref_model}")
    print(f"compound_model={args.compound_model}")
    print(f"complex_model={args.complex_model}")
    print(f"cc_model={args.cc_model}")
    print(f"relation_model={args.relation_model}")
    print(f"batch_size={args.batch_size}")
    print(f"bert_model_name={args.bert_model_name}")
    print(f"embedding={'enabled' if embedder.enabled else 'disabled'}")
    print(f"colab_light={args.colab_light}")

    for _, row in df_abs.iterrows():
        doc_id = row["doc_id"]
        abstract = row["abstract"]

        pairs = tokens_with_indices(abstract, keep_punct=True)
        tokenized_text = format_pairs(pairs)

        coref_prompt = build_coref_prompt(tokenized_text, lang=lang)
        coref_out = get_llm(args.coref_model).generate_batch(
            prompts=[coref_prompt],
            max_new_tokens=1024,
            batch_size=1,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            return_full_text=False,
        )[0][0]["generated_text"].strip()

        corefs = parse_coref_output(coref_out)
        switched = apply_coref_switch(tokenized_text, corefs)

        print(f"\n[DOC {doc_id}] ==============================")
        print(f"[DOC {doc_id}] lang={lang} chars={len(abstract)}")
        print(f"[DOC {doc_id}] Tokens: {len(pairs)} | Coref spans: {len(corefs)}")
        print(f"[DOC {doc_id}] Switched snippet: {switched[:200]}...")

        sentences = split_sentences(switched, lang=lang, embedder=embedder)
        print(f"[DOC {doc_id}] Sentences: {len(sentences)}")
        if sentences:
            print(f"[DOC {doc_id}] Sentence[0]: {sentences[0]}")

        preds = bert.predict_batch(sentences, batch_size=32)
        cls_df = pd.DataFrame(preds)

        simple_sents = cls_df[cls_df["predicted_label"] == "Simple Sentence"]["sentence"].tolist()
        compound_sents = cls_df[cls_df["predicted_label"] == "Compound Sentence"]["sentence"].tolist()
        complex_sents = cls_df[cls_df["predicted_label"] == "Complex Sentence"]["sentence"].tolist()
        cc_sents = cls_df[cls_df["predicted_label"] == "Compound-Complex Sentence"]["sentence"].tolist()

        print(
            f"[DOC {doc_id}] Label counts → "
            f"Simple={len(simple_sents)} | Compound={len(compound_sents)} | "
            f"Complex={len(complex_sents)} | Compound-Complex={len(cc_sents)}"
        )

        if sentences and (len(simple_sents) + len(compound_sents) + len(complex_sents) + len(cc_sents)) != len(
            sentences
        ):
            print(f"[DOC {doc_id}] WARNING: label split count != total sentences")

        simplified_rows = []
        if compound_sents:
            simplified_rows += simplify_sentences_for_class(
                get_llm(args.compound_model),
                "Compound Sentence",
                compound_sents,
                args.batch_size,
                lang=lang,
            )
        if complex_sents:
            simplified_rows += simplify_sentences_for_class(
                get_llm(args.complex_model),
                "Complex Sentence",
                complex_sents,
                args.batch_size,
                lang=lang,
            )
        if cc_sents:
            simplified_rows += simplify_sentences_for_class(
                get_llm(args.cc_model),
                "Compound-Complex Sentence",
                cc_sents,
                args.batch_size,
                lang=lang,
            )

        converted_simple = []
        for r in simplified_rows:
            converted_simple.extend(r.get("simplified_sentences") or [])

        final_simple_sentences = simple_sents + converted_simple

        print(
            f"[DOC {doc_id}] Relation extractor input simple sentences: "
            f"{len(final_simple_sentences)}"
        )

        rel_rows = extract_relationships(
            get_llm(args.relation_model),
            final_simple_sentences,
            batch_size=args.batch_size,
            lang=lang,
        )

        for sent_idx, rr in enumerate(rel_rows):
            sent = rr["sentence"]
            triples = rr.get("triples", []) or []
            for t in triples:
                all_rows.append({
                    "doc_id": doc_id,
                    "lang": lang,
                    "simple_sentence_index": sent_idx,
                    "simple_sentence": sent,
                    "entity_1": t.get("Entity 1"),
                    "relationship": t.get("Relationship"),
                    "entity_2": t.get("Entity 2"),
                })

    out_df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote CSV: {args.out_csv} | rows={len(out_df)}")


if __name__ == "__main__":
    main()
