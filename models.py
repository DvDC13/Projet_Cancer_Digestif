from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (recall_score, accuracy_score, precision_score, f1_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from preprocessing import build_preprocessor

MODEL_CONFIGS = {
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [None, 10, 20],
        },
        "requires_encoding": False
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "params": {
            "clf__C": [0.01, 0.1, 1.0, 10.0]
        },
        "requires_encoding": True
    },
    "SVM": {
        "model": SVC(probability=True, class_weight="balanced"),
        "params": {
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["rbf", "linear"]
        },
        "requires_encoding": True
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(),
        "params": {
            "clf__n_estimators": [50, 100],
            "clf__learning_rate": [0.05, 0.1]
        },
        "requires_encoding": False
    }
}

def train_single_model(
    name,
    config,
    X_train,
    y_train,
    X_test,
    y_test,
    selected_scoring,
    cat_cols,
    cont_cols,
    binary_cols
):
    print(f"Training {name}...")

    preprocessor = build_preprocessor(
        cat_cols,
        cont_cols,
        binary_cols,
        config["requires_encoding"]
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("clf", config["model"])
    ])

    grid = GridSearchCV(
        pipeline,
        config["params"],
        cv=10,
        scoring=selected_scoring,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    if selected_scoring == "recall_macro":
        score = recall_score(y_test, y_pred, average="macro", zero_division=0)
    elif selected_scoring == "accuracy":
        score = accuracy_score(y_test, y_pred)
    elif selected_scoring == "precision_macro":
        score = precision_score(y_test, y_pred, average="macro", zero_division=0)
    elif selected_scoring == "f1_macro":
        score = f1_score(y_test, y_pred, average="macro", zero_division=0)

    return score, grid.best_estimator_