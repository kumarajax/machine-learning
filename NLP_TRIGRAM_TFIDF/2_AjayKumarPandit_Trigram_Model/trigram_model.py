#!/usr/bin/env python3
"""Trigram language model for sentence comparison."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import pandas as pd


TEST_SENTENCES = [
    "Having the best day ever with amazing friends!",
    "Feeling so frustrated with everything going wrong today.",
]


def tokenize(text: str) -> list[str]:
    return ["<s>", "<s>"] + str(text).lower().split() + ["</s>"]


def build_counts(texts: list[str]) -> tuple[Counter[tuple[str, str, str]], Counter[tuple[str, str]], set[str]]:
    trigram_counts: Counter[tuple[str, str, str]] = Counter()
    bigram_counts: Counter[tuple[str, str]] = Counter()
    vocabulary: set[str] = set()

    for text in texts:
        tokens = tokenize(text)
        vocabulary.update(tokens)
        for index in range(len(tokens) - 2):
            bigram_counts[(tokens[index], tokens[index + 1])] += 1
            trigram_counts[(tokens[index], tokens[index + 1], tokens[index + 2])] += 1

    return trigram_counts, bigram_counts, vocabulary


def sentence_log_probability(
    sentence: str,
    trigram_counts: Counter[tuple[str, str, str]],
    bigram_counts: Counter[tuple[str, str]],
    vocabulary_size: int,
) -> float:
    tokens = tokenize(sentence)
    score = 0.0
    for index in range(len(tokens) - 2):
        w1, w2, w3 = tokens[index], tokens[index + 1], tokens[index + 2]
        numerator = trigram_counts[(w1, w2, w3)] + 1
        denominator = bigram_counts[(w1, w2)] + vocabulary_size
        score += math.log(numerator / denominator)
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/preprocessed_text.csv")
    parser.add_argument("--text-column", default="lemmatized_text")
    parser.add_argument("--output", default="outputs/trigram_sentence_scores.txt")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    texts = df[args.text_column].fillna("").astype(str).tolist()
    print(f"Loaded {len(texts)} preprocessed texts from {args.input}")

    trigram_counts, bigram_counts, vocabulary = build_counts(texts)
    vocabulary_size = len(vocabulary)
    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Trigram count: {len(trigram_counts)}")

    scored = [
        (sentence, sentence_log_probability(sentence, trigram_counts, bigram_counts, vocabulary_size))
        for sentence in TEST_SENTENCES
    ]
    winner = max(scored, key=lambda item: item[1])

    lines = ["Trigram sentence comparison", ""]
    for sentence, score in scored:
        lines.append(f"Sentence: {sentence}")
        lines.append(f"Log probability: {score:.6f}")
        lines.append("")
    lines.append(f"Recommended sentence: {winner[0]}")
    lines.append("Reason: it has the higher trigram log probability under the training corpus.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"Saved trigram result to {output_path}")


if __name__ == "__main__":
    main()

