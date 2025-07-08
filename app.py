import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import (recall_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

st.title("🔬 Prédiction de Complications Chirurgicales")

@st.cache_data
def load_data():
    df = pd.read_excel("AI_CHIR_DIG_FINAL.xlsx", sheet_name="Second cleanup")
    return df

df_raw = load_data()

# Définir les plages
input_range = df_raw.columns   # colonnes 3 à 41 (index 2 à 40)
output_range = df_raw.columns[39:73] # colonnes 42 à 74 (index 41 à 73)

st.markdown("### 🎯 Choisissez les colonnes pour l'entraînement")

selected_inputs = st.multiselect("🧮 Colonnes d'entrée (features)", input_range, default=[])
selected_target = st.selectbox("🏷️ Colonne cible (target)", options=[""] + list(output_range))

if selected_target == "" or len(selected_inputs) == 0:
    st.warning("Veuillez sélectionner au moins une colonne d'entrée et une colonne cible.")
    st.stop()

df_model = df_raw.dropna(subset=selected_inputs + [selected_target]).reset_index(drop=True)

X = df_model[selected_inputs]
y = df_model[selected_target].values.ravel().astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

model_configs = {
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

results = []
pipelines = {}

# Pre-compute column groups
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Identify binary vs continuous numericals
binary_cols = [col for col in num_cols if X_train[col].nunique() == 2]
cont_cols   = [col for col in num_cols if X_train[col].nunique() > 2]

for name, config in model_configs.items():
    print(f"Training {name}...")

    # Categorical transformer
    if config["requires_encoding"]:
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
    else:
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])

    # Continuous numerical transformer (impute then scale)
    cont_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Assemble ColumnTransformer
    preprocessor = ColumnTransformer([
        ("cat",    cat_transformer,  cat_cols),
        ("cont",   cont_transformer, cont_cols),
        ("binary","passthrough",     binary_cols),
    ], remainder="drop")  # we've explicitly handled all columns

    # Full pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("clf",          config["model"])
    ])

    # Grid search
    grid = GridSearchCV(
        pipeline,
        config["params"],
        cv=10,
        scoring="recall_macro",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)

    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=1)
    recall_per_class = recall_score(y_test, y_pred, labels=[0, 1],average=None, zero_division=1)

    results.append((name, recall_macro, recall_per_class, grid.best_estimator_))

    pipelines[name] = grid.best_estimator_

# Pick best
results.sort(key=lambda x: x[1], reverse=True)
best_model_name, best_recall_macro, best_recall_per_class, best_pipeline = results[0]

print(f"\n✅ Best model: {best_model_name}")
print(f"Macro Recall: {best_recall_macro:.4f}")
for idx, r in enumerate(best_recall_per_class):
    print(f"Recall for class {idx}: {r:.4f}")

st.markdown(f"### ✅ Meilleur modèle : **{best_model_name}** (Recall = {best_recall_per_class[1]:.2f})")

st.markdown("### 🧾 Entrez les valeurs pour prédire une complication")

user_input = {}

for col in selected_inputs:
    if pd.api.types.is_numeric_dtype(df_model[col]):
        col_min = float(df_model[col].min())
        col_max = float(df_model[col].max())
        col_median = float(df_model[col].median())
        user_input[col] = st.number_input(
            label=col,
            min_value=col_min,
            max_value=col_max,
            value=col_median
        )
    else:
        unique_values = df_model[col].dropna().unique().tolist()
        default_value = unique_values[0] if unique_values else ""
        user_input[col] = st.selectbox(
            label=col,
            options=unique_values,
            index=0 if default_value in unique_values else 0
        )

if st.button("📊 Prédire"):
    input_df = pd.DataFrame([user_input])
    pred = best_pipeline.predict(input_df)[0]
    proba = None
    if hasattr(best_pipeline.named_steps['clf'], "predict_proba"):
        proba = best_pipeline.predict_proba(input_df)[0][1]

    if proba is not None:
        st.write(f"Probabilité de complication : **{proba:.2f}**")

    # Matrice de confusion
    y_pred_test = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    st.markdown("### 📉 Matrice de confusion sur le test")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pas de complication", "Complication"])
    disp.plot(ax=ax, cmap="Blues")
    st.pyplot(fig)
