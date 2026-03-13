# Fake News Detector

A machine-learning notebook that classifies news articles as **FAKE** or **REAL** using a
TF-IDF + Passive Aggressive Classifier pipeline built with scikit-learn.

## Quick Start

1. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib joblib
   ```

2. **Add the dataset**  
   Place `news_datasets.csv` in the `data/` folder:
   ```
   data/news_datasets.csv
   ```
   The CSV must contain at least a `text` column and a `label` column (`FAKE` / `REAL`).

3. **Run the notebook**  
   Open `FAKE NEWS VERIFIER.ipynb` in Jupyter and run all cells.

## Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Setup & Imports | Import all required libraries |
| 2 | Configuration | `DATA_PATH`, `RANDOM_STATE`, and model hyper-parameters |
| 3 | Data Loading | Loads CSV with fallback to Kaggle path; validates required columns |
| 4 | Exploratory Data Analysis | Shape, null values, label distribution |
| 5 | Preprocessing | Drop nulls, stratified train/test split |
| 6 | Build Pipeline | `TfidfVectorizer` → `PassiveAggressiveClassifier` in one `Pipeline` |
| 7 | Training | `pipeline.fit(X_train, y_train)` |
| 8 | Evaluation | Accuracy, classification report, confusion matrix |
| 9 | Inference | Predict FAKE / REAL for any user-supplied text |
| 10 | Save & Load | Persist model to `models/fake_news_pipeline.joblib` |

## Dataset

This notebook was originally designed for the
[Fake News with Python](https://www.kaggle.com/datasets/hassanamin/textdb3)
dataset available on Kaggle. Download `news_datasets.csv` from there and place it in
the `data/` folder.

## Results (example)

| Metric | Score |
|--------|-------|
| Accuracy | ~93 % |
| F1 (FAKE) | ~0.93 |
| F1 (REAL) | ~0.93 |

*(Exact numbers depend on the dataset version.)*