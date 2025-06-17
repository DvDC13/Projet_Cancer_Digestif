from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

import streamlit as st

df = pd.read_excel("AI_CHIR_DIG_FINAL.xlsx", sheet_name="Second cleanup", skiprows=2)
df = df.replace('-', np.nan).dropna(subset=['COMPLICATION'])  # garde les lignes avec complication
df = df.reset_index(drop=True)

input_cols = ['AGE_SURGERY', 'SEX', 'PHX_CVD', 'RTX_PREOP', 'BMI']
target_col = 'COMPLICATION'

for col in input_cols + [target_col]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=input_cols + [target_col]).reset_index(drop=True)

X = df[input_cols]
y = df[target_col].astype(int)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
    "SVM (RBF)": SVC(kernel='rbf', class_weight='balanced'),
}

results = []
pipelines = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    results.append((name, acc, prec, rec, f1, cm))
    pipelines[name] = pipeline

# Trier par F1 Score
results.sort(key=lambda x: x[4], reverse=True)

best_model_name = results[0][0]
best_pipeline = pipelines[best_model_name]

st.title("Prédiction Complications Chirurgicales")

st.markdown("### Saisir les valeurs des variables")

user_input = {}
for col in input_cols:
    median_val = float(df[col].median())
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    user_input[col] = st.number_input(f"{col}", value=median_val, min_value=min_val, max_value=max_val)

if st.button("Prédire"):
    input_df = pd.DataFrame([user_input])
    pred = best_pipeline.predict(input_df)[0]
    proba = best_pipeline.predict_proba(input_df)[0][1] if hasattr(best_pipeline.named_steps['clf'], "predict_proba") else None

    st.write(f"### Modèle sélectionné : {best_model_name}")
    if pred == 0:
        st.error("Prediction : Complication")
    else:
        st.success("Prediction : Pas de complication")
    if proba is not None:
        st.write(f"Probabilité de complication: {proba:.2f}")

    # Afficher matrice de confusion
    y_pred_test = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    st.write("### Matrice de confusion sur le test")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pas de complication", "Complication"])
    disp.plot(ax=ax, cmap="Blues")
    st.pyplot(fig)

    # Afficher rapport classification
    st.write("### Rapport classification sur le test")
    report = classification_report(y_test, y_pred_test, target_names=["Pas de complication", "Complication"])
    st.text(report)