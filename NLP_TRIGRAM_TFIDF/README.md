# NLP Trigram and TF-IDF Assignment

Course: Natural Language Processing (S1-26_AIMLZG530)
Assignment: Assignment-1 Set-15
Group: Group 142

## How to Run

1. Go to the project folder.

```bash
cd /Users/ajay/DATA/PERSONAL/BITS/SEMESTER_2/NLP/Assignments/nlp_assignment_set15
```

2. Create the Python virtual environment.

```bash
python3 -m venv .venv
```

3. Activate the virtual environment.

```bash
source .venv/bin/activate
```

4. Install dependencies.

```bash
python3 -m pip install -r requirements.txt
```

5. Place the two CSV files in the `data/` folder.

Expected files:

```text
data/Reddit_Data.csv
data/Twitter_Data.csv
```

6. Preprocess the Reddit dataset.

```bash
python3 1_Person1_Preprocessing/preprocess.py \
  --input data/Reddit_Data.csv \
  --text-column clean_comment \
  --output outputs/reddit_preprocessed_text.csv
```

7. Preprocess the Twitter dataset.

```bash
python3 1_Person1_Preprocessing/preprocess.py \
  --input data/Twitter_Data.csv \
  --text-column clean_text \
  --output outputs/twitter_preprocessed_text.csv
```

8. Combine both preprocessed datasets into one corpus.

```bash
python3 1_Person1_Preprocessing/combine_preprocessed.py
```

9. Run the trigram language model.

```bash
python3 2_AjayKumarPandit_Trigram_Model/trigram_model.py
```

10. Run TF-IDF similarity analysis.

```bash
python3 3_Person3_TFIDF_Similarity/tfidf_similarity.py
```

11. Run PCA visualization.

```bash
python3 4_Person4_Visualization_Integration/pca_visualization.py
```

12. Check the generated outputs.

```bash
ls -lh outputs
```

Expected important files:

```text
outputs/reddit_preprocessed_text.csv
outputs/twitter_preprocessed_text.csv
outputs/preprocessed_text.csv
outputs/trigram_sentence_scores.txt
outputs/tfidf_similarity.txt
outputs/pca_tfidf_visualization.png
```

## Notes

- The large CSV data files and generated preprocessed CSV outputs are not required in GitHub if the repository is used only for source code review.
- The notebook submitted to Taxila should include visible cell outputs.
