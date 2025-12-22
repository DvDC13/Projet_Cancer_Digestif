import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)
from sklearn.pipeline import Pipeline
from itertools import combinations

import preprocessing


def evaluate_model_on_inputs(
    model,
    requires_encoding,
    X,
    y,
    scoring
):
    """
    Evaluate a model using ONLY the provided input columns.
    Returns a single scalar score.
    """

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=42,
        stratify=y
    )

    # Split columns
    cat_cols, cont_cols, binary_cols = preprocessing.split_columns(X_train)

    # Build preprocessing
    preprocessor = preprocessing.build_preprocessor(
        cat_cols,
        cont_cols,
        binary_cols,
        requires_encoding
    )

    # Full pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("clf", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # =============================
    # Metric handling (FIX HERE)
    # =============================

    if scoring == "Accuracy":
        return accuracy_score(y_test, y_pred)

    elif scoring == "Recall (macro)":
        return recall_score(y_test, y_pred, average="macro", zero_division=0)

    elif scoring == "Precision (macro)":
        return precision_score(y_test, y_pred, average="macro", zero_division=0)

    elif scoring == "F1-score (macro)":
        return f1_score(y_test, y_pred, average="macro", zero_division=0)

    else:
        raise ValueError(f"Unknown scoring metric: {scoring}")

def evaluate_by_input_count(
    model,
    requires_encoding,
    X,
    y,
    scoring,
    max_inputs=4
):
    """
    Evaluate mean performance using 1..max_inputs variables
    """

    results = {}

    input_cols = list(X.columns)

    for k in range(1, max_inputs + 1):
        scores = []

        for cols in combinations(input_cols, k):
            X_subset = X[list(cols)]

            score = evaluate_model_on_inputs(
                model=model,
                requires_encoding=requires_encoding,
                X=X_subset,
                y=y,
                scoring=scoring
            )

            scores.append(score)

        results[k] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "n_combinations": len(scores)
        }

    return results