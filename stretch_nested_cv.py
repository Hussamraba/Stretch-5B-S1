import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.base import clone


# ---------------------------------------------------
# 1) Load data
# ---------------------------------------------------

NUMERIC_FEATURES = [
    "senior_citizen",
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
]


def load_data(filepath="data/telecom_churn.csv"):
    df = pd.read_csv(filepath)

    X = df[NUMERIC_FEATURES].copy()
    y = df["churned"].copy()

    return X, y


# ---------------------------------------------------
# 2) Part 1: GridSearchCV for Random Forest
# ---------------------------------------------------

def run_rf_grid_search(X, y):
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )

    grid.fit(X, y)
    return grid


def save_rf_heatmap(grid, outpath="rf_grid_heatmap.png"):
    results = pd.DataFrame(grid.cv_results_)

    best_split = grid.best_params_["min_samples_split"]

    filtered = results[
        results["param_min_samples_split"] == best_split
    ].copy()

    pivot = filtered.pivot_table(
        index="param_max_depth",
        columns="param_n_estimators",
        values="mean_test_score",
    )

    pivot = pivot.reindex([3, 5, 10, 20, None])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])

    ax.set_xlabel("n_estimators")
    ax.set_ylabel("max_depth")
    ax.set_title(
        f"Random Forest GridSearchCV F1 Heatmap\n"
        f"(min_samples_split fixed at best={best_split})"
    )

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, label="Mean CV F1")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# ---------------------------------------------------
# 3) Part 2: Nested Cross-Validation
# ---------------------------------------------------

def nested_cv_scores(model, param_grid, X, y, inner_random_state=42, outer_random_state=99):
    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=inner_random_state,
    )

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=outer_random_state,
    )

    outer_scores = []
    inner_best_scores = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]

        grid = GridSearchCV(
            estimator=clone(model),
            param_grid=param_grid,
            scoring="f1",
            cv=inner_cv,
            n_jobs=-1,
        )

        grid.fit(X_train_fold, y_train_fold)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test_fold)
        outer_f1 = f1_score(y_test_fold, y_pred)

        inner_best_scores.append(grid.best_score_)
        outer_scores.append(outer_f1)

    return {
        "inner_scores": inner_best_scores,
        "outer_scores": outer_scores,
        "inner_mean": float(np.mean(inner_best_scores)),
        "outer_mean": float(np.mean(outer_scores)),
        "gap": float(np.mean(inner_best_scores) - np.mean(outer_scores)),
    }


def build_nested_cv_comparison_table(X, y):
    rf_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    rf_param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
    }

    dt_model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
    )

    dt_param_grid = {
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
    }

    rf_results = nested_cv_scores(
        model=rf_model,
        param_grid=rf_param_grid,
        X=X,
        y=y,
        inner_random_state=42,
        outer_random_state=99,
    )

    dt_results = nested_cv_scores(
        model=dt_model,
        param_grid=dt_param_grid,
        X=X,
        y=y,
        inner_random_state=42,
        outer_random_state=99,
    )

    comparison = pd.DataFrame({
        "Metric": [
            "Inner best_score_ (mean across outer folds)",
            "Outer nested CV score (mean across outer folds)",
            "Gap (inner - outer)",
        ],
        "Random Forest": [
            rf_results["inner_mean"],
            rf_results["outer_mean"],
            rf_results["gap"],
        ],
        "Decision Tree": [
            dt_results["inner_mean"],
            dt_results["outer_mean"],
            dt_results["gap"],
        ],
    })

    return comparison, rf_results, dt_results


# ---------------------------------------------------
# 4) Main
# ---------------------------------------------------

def main():
    X, y = load_data()

    # Part 1
    rf_grid = run_rf_grid_search(X, y)
    save_rf_heatmap(rf_grid, outpath="rf_grid_heatmap.png")

    print("=== Part 1: Random Forest Grid Search ===")
    print("Best params:", rf_grid.best_params_)
    print("Best CV F1:", round(rf_grid.best_score_, 4))
    print()

    cv_results = pd.DataFrame(rf_grid.cv_results_)
    cv_results.to_csv("rf_grid_results.csv", index=False)

    # Part 2
    comparison_table, rf_nested, dt_nested = build_nested_cv_comparison_table(X, y)

    print("=== Part 2: Nested CV Comparison ===")
    print(comparison_table)
    print()

    comparison_table.to_csv("nested_cv_comparison.csv", index=False)

    print("Random Forest inner scores:", [round(x, 4) for x in rf_nested["inner_scores"]])
    print("Random Forest outer scores:", [round(x, 4) for x in rf_nested["outer_scores"]])
    print()

    print("Decision Tree inner scores:", [round(x, 4) for x in dt_nested["inner_scores"]])
    print("Decision Tree outer scores:", [round(x, 4) for x in dt_nested["outer_scores"]])


if __name__ == "__main__":
    main()