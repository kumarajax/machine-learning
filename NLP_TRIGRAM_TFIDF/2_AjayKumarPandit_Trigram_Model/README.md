# Person 2: Trigram Language Model

Goal: complete Part I sentence comparison.

Responsibilities:

- Use the full dataset text as the training corpus.
- Build unigram, bigram, and trigram counts.
- Score both assignment test sentences using trigram probabilities.
- Use smoothing and log probabilities.
- Write a short conclusion recommending the more probable sentence.

Run from the assignment root after Person 1 creates `outputs/preprocessed_text.csv`:

```bash
python3 Person2_Trigram_Model/trigram_model.py
```

