#!/usr/bin/env python3
"""Preprocess text for NLP Assignment-1 Set-15."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


TEXT_COLUMN_CANDIDATES = [
    "text",
    "comment",
    "clean_comment",
    "tweet",
    "sentence",
    "content",
    "body",
]


def ensure_nltk() -> None:
    for package in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(package, quiet=True)


def infer_text_column(df: pd.DataFrame) -> str:
    lower_to_original = {column.lower(): column for column in df.columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    object_columns = [column for column in df.columns if df[column].dtype == "object"]
    if not object_columns:
        raise ValueError("No text-like column found. Pass --text-column explicitly.")

    return max(object_columns, key=lambda column: df[column].astype(str).str.len().mean())


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str, stop_words: set[str], stemmer: PorterStemmer, lemmatizer: WordNetLemmatizer) -> dict[str, str]:
    normalized = normalize_text(text)
    tokens = [token for token in word_tokenize(normalized) if token.isalpha()]
    without_stopwords = [token for token in tokens if token not in stop_words]
    stems = [stemmer.stem(token) for token in without_stopwords]
    lemmas = [lemmatizer.lemmatize(token, wordnet.NOUN) for token in without_stopwords]

    return {
        "normalized_text": normalized,
        "tokens": " ".join(tokens),
        "without_stopwords": " ".join(without_stopwords),
        "stemmed_text": " ".join(stems),
        "lemmatized_text": " ".join(lemmas),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--text-column", help="Text column name. Auto-detected if omitted.")
    parser.add_argument("--output", default="outputs/preprocessed_text.csv")
    args = parser.parse_args()

    ensure_nltk()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    text_column = args.text_column or infer_text_column(df)
    print(f"Loaded {len(df)} rows from {input_path}")
    print(f"Using text column: {text_column}")

    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    processed_rows = df[text_column].fillna("").apply(
        lambda value: preprocess_text(value, stop_words, stemmer, lemmatizer)
    )
    processed_df = pd.DataFrame(processed_rows.tolist())
    result = pd.concat([df.reset_index(drop=True), processed_df], axis=1)
    result.to_csv(output_path, index=False)

    print(f"Saved preprocessed data to {output_path}")


if __name__ == "__main__":
    main()
