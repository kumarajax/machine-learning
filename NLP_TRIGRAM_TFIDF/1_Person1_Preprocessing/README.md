# Person 1: Dataset Loading and Text Preprocessing

Goal: complete Part II preprocessing.

Responsibilities:

- Load the dataset from `../data/`.
- Identify the text column.
- Apply tokenization, lowercasing, stopword removal, stemming, and lemmatization.
- Save the processed output to `../outputs/preprocessed_text.csv`.

Run from the assignment root:

```bash
python3 Person1_Preprocessing/preprocess.py --input data/<file>.csv
```

If needed:

```bash
python3 Person1_Preprocessing/preprocess.py --input data/<file>.csv --text-column clean_comment
```

For the assignment dataset, run both files:

```bash
python3 Person1_Preprocessing/preprocess.py --input data/Reddit_Data.csv --text-column clean_comment --output outputs/reddit_preprocessed_text.csv
python3 Person1_Preprocessing/preprocess.py --input data/Twitter_Data.csv --text-column clean_text --output outputs/twitter_preprocessed_text.csv
python3 Person1_Preprocessing/combine_preprocessed.py
```
