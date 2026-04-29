"""
Main orchestrator: prepare all training data for EfficientRAG.

Combines three data sources:
1. HotpotQA (EN) — free, heuristic-based token labels, no LLM
2. Dragon-derec single-hop (RU) — free, exact-match token labels
3. Dragon-derec multi-hop (RU) — ~$1 LLM cost for synthesis

Final output:
    data/efficient_rag/labeler/train.jsonl
    data/efficient_rag/filter/train.jsonl

Usage:
    python -m EfficientRAG.data_synthesis.prepare_training_data \
        --hotpotqa_samples 5000 \
        --dragon_multi_hop \
        --llm_model meta-llama/llama-3.1-8b-instruct \
        --output_dir data/efficient_rag
"""

import argparse
import json
import logging
import os
import random

logger = logging.getLogger(__name__)


def prepare_all(
    output_dir: str = "data/efficient_rag",
    hotpotqa_samples: int = 5000,
    dragon_multi_hop: bool = True,
    llm_fn=None,
    llm_model: str = "meta-llama/llama-3.1-8b-instruct",
    api_key: str = None,
    api_base: str = "https://openrouter.ai/api/v1",
    corpus_path: str = "dragon_corpus.jsonl",
    seed: int = 42,
):
    """
    Prepare all training data.

    Args:
        output_dir: output directory
        hotpotqa_samples: number of HotpotQA samples (0 to skip)
        dragon_multi_hop: whether to synthesize dragon multi-hop data (needs LLM)
        llm_fn: optional LLM callable (if None, creates from api_key)
        llm_model: model name for OpenRouter
        api_key: OpenRouter API key
        api_base: API base URL
        corpus_path: path to dragon_corpus.jsonl
        seed: random seed
    """
    os.makedirs(output_dir, exist_ok=True)

    all_labeler = []
    all_filter = []

    # --- 1. HotpotQA (EN, free) ---
    if hotpotqa_samples > 0:
        logger.info(f"=== Step 1: HotpotQA ({hotpotqa_samples} samples) ===")
        from EfficientRAG.data_synthesis.download_hotpotqa_data import (
            generate_hotpotqa_labeler_data,
        )

        hotpot_labeler, hotpot_filter = generate_hotpotqa_labeler_data(
            num_samples=hotpotqa_samples,
            output_dir=output_dir,
            seed=seed,
        )
        all_labeler.extend(hotpot_labeler)
        all_filter.extend(hotpot_filter)
        logger.info(
            f"HotpotQA: {len(hotpot_labeler)} labeler, {len(hotpot_filter)} filter"
        )

    # --- 2. Dragon single-hop (RU, free) ---
    logger.info("=== Step 2: Dragon single-hop (free) ===")
    from datasets import load_dataset

    from EfficientRAG.data_synthesis.dragon_single_hop import process_and_save

    dragon_ds = load_dataset("Makson4ic/dragon-derec-dataset")
    dragon_pos, dragon_neg = process_and_save(
        dataset_split=dragon_ds["train"],
        corpus_path=corpus_path,
        output_dir=output_dir,
        seed=seed,
    )
    all_labeler.extend(dragon_pos)
    all_labeler.extend(dragon_neg)
    logger.info(
        f"Dragon single-hop: {len(dragon_pos)} FINISH + {len(dragon_neg)} TERMINATE"
    )

    # --- 3. Dragon multi-hop (RU, needs LLM) ---
    if dragon_multi_hop:
        logger.info("=== Step 3: Dragon multi-hop (LLM synthesis) ===")

        if llm_fn is None:
            if api_key is None:
                from dotenv import load_dotenv

                load_dotenv()
                api_key = os.getenv("OPENAI_API_KEY")

            if api_key:
                from openai import OpenAI

                client = OpenAI(api_key=api_key, base_url=api_base)

                def llm_fn(prompt: str) -> str:
                    resp = client.chat.completions.create(
                        model=llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=1024,
                    )
                    return resp.choices[0].message.content
            else:
                logger.warning(
                    "No API key found, skipping multi-hop synthesis. "
                    "Set OPENAI_API_KEY in .env"
                )
                llm_fn = None

        if llm_fn is not None:
            # Get multi-hop samples
            train_grounded = [
                s for s in dragon_ds["train"] if s["is_grounded"]
            ]
            multi_hop = [
                s
                for s in train_grounded
                if len(set(s["found_ids"])) >= 2
            ]
            logger.info(f"Multi-hop samples to process: {len(multi_hop)}")

            # Format for decomposition
            formatted = []
            for s in multi_hop:
                ref = s["reference_answer"]
                try:
                    parsed = json.loads(ref)
                    if isinstance(parsed, list):
                        ref = parsed[0] if parsed else ref
                except (json.JSONDecodeError, TypeError):
                    pass

                seen = set()
                paragraphs = []
                for fid, text in zip(s["found_ids"], s["evidence_texts"]):
                    if fid not in seen:
                        seen.add(fid)
                        paragraphs.append(
                            {"id": str(fid), "title": f"Документ {fid}", "text": text}
                        )

                formatted.append(
                    {
                        "id": s["question_id"],
                        "question": s["question"],
                        "answer": ref,
                        "supporting_paragraphs": paragraphs,
                    }
                )

            # Step 3a: Decompose
            from EfficientRAG.data_synthesis.query_decompose import decompose_dataset

            decomposed = decompose_dataset(
                dataset=formatted,
                llm_fn=llm_fn,
                dataset_name="dragon-derec",
                output_path=os.path.join(output_dir, "dragon_decomposed.jsonl"),
            )

            # Step 3b: Token labeling
            from EfficientRAG.data_synthesis.token_labeling import (
                label_tokens_dataset,
            )

            token_labeled = label_tokens_dataset(
                decomposed_data=decomposed,
                llm_fn=llm_fn,
                spacy_model="ru_core_news_sm",
                output_path=os.path.join(output_dir, "dragon_token_labeled.jsonl"),
            )

            # Step 3c: Filter data
            from EfficientRAG.data_synthesis.next_hop_query_filtering import (
                construct_filter_dataset,
            )

            filter_samples = construct_filter_dataset(
                token_labeled_data=token_labeled,
                llm_fn=llm_fn,
                spacy_model="ru_core_news_sm",
            )
            all_filter.extend(filter_samples)

            # Step 3d: Build labeler samples from multi-hop
            from EfficientRAG.data_synthesis.training_data_synthesize import (
                build_labeler_data,
            )

            # Negative sampling from corpus
            from EfficientRAG.data_synthesis.negative_sampling import (
                build_negative_samples,
            )

            corpus = []
            if os.path.exists(corpus_path):
                with open(corpus_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            corpus.append(json.loads(line))

            rng = random.Random(seed)

            def corpus_retriever(query, top_k):
                qw = set(query.lower().split())
                scored = []
                for doc in corpus:
                    tw = set(doc.get("text", "").lower().split())
                    scored.append((len(qw & tw), doc))
                scored.sort(key=lambda x: -x[0])
                return [
                    {"text": d["text"], "id": d.get("id", ""), "score": s}
                    for s, d in scored[:top_k]
                ]

            neg_samples = build_negative_samples(
                token_labeled_data=token_labeled,
                retriever_fn=corpus_retriever,
            )

            multi_labeler = build_labeler_data(
                token_labeled_data=token_labeled,
                negative_samples=neg_samples,
            )
            all_labeler.extend(multi_labeler)

            logger.info(
                f"Dragon multi-hop: {len(multi_labeler)} labeler, "
                f"{len(filter_samples)} filter"
            )

    # --- 4. Shuffle and save final files ---
    rng = random.Random(seed)
    rng.shuffle(all_labeler)
    rng.shuffle(all_filter)

    labeler_dir = os.path.join(output_dir, "labeler")
    filter_dir = os.path.join(output_dir, "filter")
    os.makedirs(labeler_dir, exist_ok=True)
    os.makedirs(filter_dir, exist_ok=True)

    lab_path = os.path.join(labeler_dir, "train.jsonl")
    with open(lab_path, "w", encoding="utf-8") as f:
        for s in all_labeler:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    fil_path = os.path.join(filter_dir, "train.jsonl")
    with open(fil_path, "w", encoding="utf-8") as f:
        for s in all_filter:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Stats
    from collections import Counter

    tag_counts = Counter(s.get("tag", "") for s in all_labeler)

    logger.info("=" * 50)
    logger.info(f"FINAL TRAINING DATA:")
    logger.info(f"  Labeler: {len(all_labeler)} samples → {lab_path}")
    for tag, count in sorted(tag_counts.items()):
        logger.info(f"    {tag}: {count}")
    logger.info(f"  Filter:  {len(all_filter)} samples → {fil_path}")
    logger.info("=" * 50)

    return all_labeler, all_filter


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Prepare EfficientRAG training data")
    parser.add_argument("--output_dir", default="data/efficient_rag")
    parser.add_argument("--hotpotqa_samples", type=int, default=5000)
    parser.add_argument(
        "--no_dragon_multi_hop", action="store_true", help="Skip multi-hop (no LLM cost)"
    )
    parser.add_argument(
        "--llm_model", default="meta-llama/llama-3.1-8b-instruct",
        help="LLM for synthesis (default: 8b, cheaper)"
    )
    parser.add_argument("--corpus_path", default="dragon_corpus.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_all(
        output_dir=args.output_dir,
        hotpotqa_samples=args.hotpotqa_samples,
        dragon_multi_hop=not args.no_dragon_multi_hop,
        llm_model=args.llm_model,
        corpus_path=args.corpus_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
