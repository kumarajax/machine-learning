#!/usr/bin/env python3
"""TF-IDF feature extraction and word similarity analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def top_similar_word_pairs(tfidf_matrix, feature_names: np.ndarray, top_n: int = 2) -> list[tuple[str, str, float]]:
    word_vectors = tfidf_matrix.T
    similarity = cosine_similarity(word_vectors)
    np.fill_diagonal(similarity, -1.0)

    pairs: list[tuple[str, str, float]] = []
    used: set[tuple[int, int]] = set()

    for flat_index in np.argsort(similarity.ravel())[::-1]:
        row, column = np.unravel_index(flat_index, similarity.shape)
        pair_key = tuple(sorted((row, column)))
        if pair_key in used:
            continue
        used.add(pair_key)
        pairs.append((feature_names[row], feature_names[column], float(similarity[row, column])))
        if len(pairs) == top_n:
            break

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/preprocessed_text.csv")
    parser.add_argument("--text-column", default="lemmatized_text")
    parser.add_argument("--output", default="outputs/tfidf_similarity.txt")
    parser.add_argument("--max-features", type=int, default=3000)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    texts = df[args.text_column].fillna("").astype(str).tolist()
    print(f"Loaded {len(texts)} texts from {args.input}")
    print("Building TF-IDF matrix...")

    vectorizer = TfidfVectorizer(max_features=args.max_features, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    print(f"TF-IDF shape: {tfidf_matrix.shape}")

    pairs = top_similar_word_pairs(tfidf_matrix, feature_names, top_n=2)

    lines = [
        "TF-IDF word similarity analysis",
        "",
        "Metric: cosine similarity",
        "Justification: cosine similarity compares direction rather than raw magnitude, making it suitable for sparse TF-IDF vectors.",
        "",
        "Top two similar word pairs:",
    ]
    for index, (word_a, word_b, score) in enumerate(pairs, start=1):
        lines.append(f"{index}. {word_a} - {word_b}: {score:.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"Saved similarity result to {output_path}")


if __name__ == "__main__":
    main()

