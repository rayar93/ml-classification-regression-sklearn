# Machine Learning Classification & Regression (scikit-learn)

Team project - CS 4440 Artificial Intelligence, Appalachian State University.

**Members:** Alan Ray · Zach Shotwell · Miguel Moreno Coin

## Overview
Two end-to-end supervised learning pipelines, submitted as the midterm project:

- **Classification** — Predicts student performance category (Average / Good / Very Good / Excellent)
  from demographic and academic features using the CEE Student Entrance Exam dataset.
  Best model: Logistic Regression, 54.4% test accuracy (25% random-chance baseline).
- **Regression** — Predicts auction verification runtime (`verification.time`) from auction process
  parameters using the Auction Verification dataset.
  Best model: Random Forest, R²=0.991, RMSE=1,078s.

Each task compares four models evaluated via 10-fold cross-validation and tuned with GridSearchCV.

## Repository structure
```
ml-classification-regression-sklearn/
├── data/
│   ├── raw/
│   │   ├── CEE_DATA.arff
│   │   └── data.csv
│   └── processed/
│       ├── CEE_DATA.csv
│       └── Auction_Verification.csv
├── notebooks/
│   └── midterm_project.ipynb
├── reports/
│   └── report.pdf
└── requirements.txt
```
## Setup

```bash
git clone https://github.com/rayar93/ml-classification-regression-sklearn
cd ml-classification-regression-sklearn
pip install -r requirements.txt
```

## Running

Open `notebooks/midterm_project.ipynb` in VSCode or JupyterLab and run all cells top to bottom. The notebook writes cleaned data to `data/processed/` and saves trained models as `.joblib` files in the `notebooks/` directory.

## Requirements

See `requirements.txt`. Key dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `joblib`.

## Report

- **Report:** [reports/report.pdf](reports/report.pdf) - the midterm write-up covering both pipelines and all eight models
