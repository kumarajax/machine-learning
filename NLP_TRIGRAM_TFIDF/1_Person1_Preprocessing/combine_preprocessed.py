#!/usr/bin/env python3
"""Combine Reddit and Twitter preprocessed files into one training corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUTS = [
    ("outputs/reddit_preprocessed_text.csv", "reddit"),
    ("outputs/twitter_preprocessed_text.csv", "twitter"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/preprocessed_text.csv")
    args = parser.parse_args()

    frames = []
    for file_name, source_name in DEFAULT_INPUTS:
        path = Path(file_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run preprocessing for {source_name} first."
            )

        frame = pd.read_csv(path)
        frame["source_dataset"] = source_name
        frames.append(frame)
        print(f"Loaded {len(frame)} rows from {path}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"Combined rows: {len(combined)}")
    print(f"Saved combined corpus to {output_path}")


if __name__ == "__main__":
    main()

