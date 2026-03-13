# FakeNewsDetecter

A basic Python script that classifies news articles as **real** or **fake** using a
**Logistic Regression** model trained on TF-IDF text features.

---

## How it works

1. **Text pre-processing** – lower-casing, removing numbers and punctuation.
2. **TF-IDF vectorisation** – converts article text into numerical feature vectors
   (unigrams + bigrams, top 10 000 features, English stop-words removed).
3. **Logistic Regression** – a fast, interpretable binary classifier.
4. **Evaluation** – accuracy, precision/recall/F1, and a confusion matrix are
   printed to the console.

---

## Requirements

- Python 3.8+
- scikit-learn ≥ 1.0
- pandas ≥ 1.3

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage

### Run on the built-in sample data

```bash
python fake_news_detector.py
```

A small set of 20 hand-crafted headlines (10 real, 10 fake) is included so you
can try the script immediately without any external data.

### Run on your own CSV dataset

```bash
python fake_news_detector.py path/to/dataset.csv
```

The CSV must contain at least these two columns:

| Column  | Description                                          |
|---------|------------------------------------------------------|
| `text`  | Article body (or headline + body concatenated)       |
| `label` | `0` / `1`  **or**  `REAL` / `FAKE` (case-insensitive) |

A popular public dataset that matches this format is the
[ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

---

## Example output

```
No dataset path provided – using built-in sample data.
Dataset size: 20 articles  (real: 10, fake: 10)

Training logistic regression model …
Test Accuracy : 0.7500

Classification Report:
              precision    recall  f1-score   support

        Real       0.67      1.00      0.80         2
        Fake       1.00      0.50      0.67         2

    accuracy                           0.75         4

Confusion Matrix (rows=actual, cols=predicted):
            Predicted Real  Predicted Fake
Actual Real       2               0
Actual Fake       1               1

Example predictions on new headlines:
------------------------------------------------------------
  [Real]  Scientists develop new battery technology ...
  [Fake]  BREAKING: Government admits chemtrails ...
```

---

## Project structure

```
FakeNewsDetecter/
├── fake_news_detector.py   # Main script
├── requirements.txt        # Python dependencies
└── README.md
```