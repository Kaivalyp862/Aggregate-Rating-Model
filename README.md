# Zomato Restaurant Rating Prediction

Predicting a restaurant's aggregate customer rating from operational and location features — built as Task 1 of the **Cognifyz Technologies Machine Learning Internship**.

## Overview

Given a restaurant's cost, cuisine mix, location, and service options, can we predict how it'll be rated? This project builds a regression pipeline over a 9,551-restaurant Zomato dataset to answer that, using structured operational features rather than the reviews themselves.

## Dataset

9,551 restaurants from Zomato, spanning multiple countries with a strong skew toward India. Fields include cost for two, cuisines, table booking / online delivery availability, price range, vote counts, and the aggregate rating being predicted.

> Add your dataset source link here.

Because the dataset mixes currencies across countries, raw cost isn't directly comparable restaurant-to-restaurant — see `Relative_Cost_City` below for how that's handled.

## Approach

**Feature engineering**
- `Is_Chain` — flags restaurants whose name appears more than once in the dataset (a chain, not a one-off)
- `Multi_Cuisines` — flags restaurants listing more than one cuisine
- `Relative_Cost_City` — cost for two, relative to that restaurant's own city average. Since a city rarely spans currencies, this sidesteps the cross-country cost comparison problem without needing explicit currency conversion.

**Preprocessing**
- Target encoding for high-cardinality categorical fields (`City`, `Locality`)
- One-hot encoding for binary fields (`Has Table booking`, `Has Online delivery`)
- Numeric features passed through unchanged

**Model**
- XGBoost Regressor (`learning_rate=0.05`, `max_depth=3`, `n_estimators=200`, `subsample=0.8`)

**Persistence**
- The trained model and fitted preprocessing pipeline are pickled separately (`model.pkl`, `pipeline.pkl`). A fresh run trains and saves both; every run after loads them from disk and skips straight to inference.

## Results

| Metric | Value |
|---|---|
| MAE  | _fill in after running_ |
| RMSE | _fill in after running_ |
| R²   | _fill in after running_ |

## Project structure

```
.
├── Final_Model.ipynb   # Feature engineering, training, evaluation
├── model.pkl           # Pickled XGBoost model (generated on first run)
├── pipeline.pkl        # Pickled preprocessing pipeline (generated on first run)
├── predictions.csv     # Actual vs. predicted ratings on the held-out set
└── README.md
```

## Getting started

```bash
pip install pandas numpy scikit-learn category_encoders xgboost
```

Place `Dataset .csv` in the project root and run the notebook top to bottom. The first run trains and pickles the model; later runs load it straight from disk and evaluate immediately.

## Notes / possible improvements

- Earlier EDA flagged restaurants with an aggregate rating of 0 (i.e. not yet rated) — worth double-checking these are excluded before training if they're still present in `Dataset .csv`, since a 0 usually means "unrated," not "rated poorly."
- Open question: keep this global, or scope it to India-only listings, which make up most of the dataset?
- `Relative_Cost_City` is currently computed on the full dataset before the train/test split. Computing city means on the training split only would remove that source of leakage.

## About

Built as part of the Cognifyz Technologies Machine Learning Internship (Task 1: Predict Restaurant Ratings).

**Kaivalya Anil Patil**
[GitHub](https://github.com/Kaivalyp862) · [LinkedIn](https://www.linkedin.com/in/kaivalya-anil-patil-67a52a305)
