#!/usr/bin/env python3
"""PCA visualization for TF-IDF vectors."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/preprocessed_text.csv")
    parser.add_argument("--text-column", default="lemmatized_text")
    parser.add_argument("--output", default="outputs/pca_tfidf_visualization.png")
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--max-features", type=int, default=1000)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    sample = df.sample(min(args.sample_size, len(df)), random_state=42)
    texts = sample[args.text_column].fillna("").astype(str).tolist()
    print(f"Using {len(texts)} sampled rows for PCA visualization")

    vectorizer = TfidfVectorizer(max_features=args.max_features, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF shape before PCA: {tfidf_matrix.shape}")

    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(tfidf_matrix.toarray())
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    plt.figure(figsize=(9, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=18, alpha=0.7)
    plt.title("2D PCA Visualization of TF-IDF Text Vectors")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    print(f"Saved PCA visualization to {output_path}")


if __name__ == "__main__":
    main()

