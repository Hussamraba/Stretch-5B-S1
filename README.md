# Module 5 Week B — Stretch: Hyperparameter Tuning & Nested Cross-Validation

## Overview

This project implements systematic hyperparameter tuning and evaluates model performance using nested cross-validation on the telecom churn dataset.

The goal is to:

* Tune a Random Forest model using GridSearchCV
* Visualize model performance across hyperparameters
* Measure selection bias using nested cross-validation
* Compare Random Forest vs Decision Tree

---

## Project Structure

```
stretch-5b-nested-cv/
│
├── stretch_nested_cv.py
├── rf_grid_heatmap.png
├── rf_grid_results.csv
├── nested_cv_comparison.csv
├── data/
│   └── telecom_churn.csv
└── README.md
```

---

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## Run

```bash
python stretch_nested_cv.py
```

---

## Part 1 — GridSearchCV

A Random Forest model was tuned using the following parameter grid:

* n_estimators: [50, 100, 200]
* max_depth: [3, 5, 10, 20, None]
* min_samples_split: [2, 5, 10]

5-fold stratified cross-validation was used with F1 score.

### Results

* Best parameters:

```
{'max_depth': 3, 'min_samples_split': 2, 'n_estimators': 100}
```

* Best CV F1:

```
0.2944
```

A heatmap (`rf_grid_heatmap.png`) shows performance across max_depth and n_estimators.

---

## Part 1 — Analysis

The most impactful hyperparameter was max_depth. Shallow trees (low max_depth) performed best, suggesting that deeper trees may lead to overfitting. Increasing the number of estimators improved performance slightly but showed diminishing returns.

The results indicate that the model benefits from stronger regularization and is more prone to overfitting at higher depths. Performance tends to plateau beyond a certain number of trees, suggesting limited gains from further increasing n_estimators.

---

## Part 2 — Nested Cross-Validation

Nested cross-validation was used to evaluate the true generalization performance and measure selection bias.

* Inner loop: GridSearchCV (5-fold)
* Outer loop: StratifiedKFold (5-fold)

Two models were evaluated:

* Random Forest (same grid as Part 1)
* Decision Tree

---

## Results

| Metric                | Random Forest | Decision Tree |
| --------------------- | ------------: | ------------: |
| Inner best_score_     |        0.3011 |        0.2739 |
| Outer nested CV score |        0.2653 |        0.2451 |
| Gap (inner - outer)   |        0.0357 |        0.0288 |

---

## Part 2 — Analysis

The Random Forest shows a slightly larger gap between inner and outer scores compared to the Decision Tree. This indicates some level of selection bias due to hyperparameter tuning.

However, both models show a noticeable drop from inner to outer performance, confirming that GridSearchCV results alone are optimistic.

Random Forest remains more stable due to averaging across multiple trees, while Decision Trees are more sensitive to data splits.

This demonstrates the importance of nested cross-validation: the outer loop acts as a true evaluation set that was not used during hyperparameter tuning, providing a more reliable estimate of real-world performance.

---

## Key Takeaways

* GridSearchCV alone can overestimate performance
* Nested CV provides a more honest evaluation
* Model complexity must be controlled to avoid overfitting
* Random Forest is generally more stable than Decision Tree

---

## Outputs

The script generates:

* `rf_grid_heatmap.png` — visualization of F1 scores
* `rf_grid_results.csv` — full grid search results
* `nested_cv_comparison.csv` — comparison of inner vs outer scores

---

## Author

Hussam Rabaa
