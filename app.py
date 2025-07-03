import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import Counter

st.title("🔬 Prédiction de Complications Chirurgicales")

@st.cache_data
def load_data():
    df = pd.read_excel("AI_CHIR_DIG_FINAL.xlsx", sheet_name="Second cleanup", skiprows=2)
    df = df.replace('-', np.nan)
    df = df.dropna(how="all")
    df = df.reset_index(drop=True)
    return df

df_raw = load_data()

# Supprimer les deux premières colonnes : PT_NUM et NAME_OPERATION
df = df_raw.iloc[:, 2:]

# Définir les plages
input_range = df.columns[0:39]   # colonnes 3 à 41 (index 2 à 40)
output_range = df.columns[39:73] # colonnes 42 à 74 (index 41 à 73)

st.markdown("### 🎯 Choisissez les colonnes pour l'entraînement")

selected_inputs = st.multiselect("🧮 Colonnes d'entrée (features)", input_range, default=[])
selected_target = st.selectbox("🏷️ Colonne cible (target)", options=[""] + list(output_range))

if selected_target == "" or len(selected_inputs) == 0:
    st.warning("Veuillez sélectionner au moins une colonne d'entrée et une colonne cible.")
    st.stop()

# Nettoyage des colonnes sélectionnées
df_model = df[selected_inputs + [selected_target]].copy()
for col in selected_inputs + [selected_target]:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

df_model = df_model.dropna(subset=selected_inputs + [selected_target]).reset_index(drop=True)

X = df_model[selected_inputs]
y = df_model[selected_target].astype(int)

class_counts = Counter(y)
min_class_size = min(class_counts.values())

# Use smaller k_neighbors for rare classes
k_neighbors = min(5, min_class_size - 1)
if k_neighbors < 1:
    st.warning("⚠️ Trop peu de données dans certaines classes pour appliquer SMOTE.")
    st.stop()

smote = SMOTE(random_state=42, k_neighbors=k_neighbors)

# SMOTE
#
# smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Modèles
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
    "SVM (RBF)": SVC(kernel='rbf', class_weight='balanced', probability=True),
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

    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    results.append((name, f1))
    pipelines[name] = pipeline

results.sort(key=lambda x: x[1], reverse=True)
best_model_name, best_f1 = results[0]
best_pipeline = pipelines[best_model_name]

st.markdown(f"### ✅ Meilleur modèle : **{best_model_name}** (F1 = {best_f1:.2f})")

st.markdown("### 🧾 Entrez les valeurs pour prédire une complication")

user_input = {}
for col in selected_inputs:
    col_min = float(df_model[col].min())
    col_max = float(df_model[col].max())
    col_median = float(df_model[col].median())
    user_input[col] = st.number_input(
        label=col,
        min_value=col_min,
        max_value=col_max,
        value=col_median
    )

if st.button("📊 Prédire"):
    input_df = pd.DataFrame([user_input])
    pred = best_pipeline.predict(input_df)[0]
    proba = None
    if hasattr(best_pipeline.named_steps['clf'], "predict_proba"):
        proba = best_pipeline.predict_proba(input_df)[0][1]

    if pred == 0:
        st.error("❌ Prédiction : Complication")
    else:
        st.success("✅ Prédiction : Pas de complication")

    if proba is not None:
        st.write(f"Probabilité de complication : **{proba:.2f}**")

    # Matrice de confusion
    y_pred_test = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    st.markdown("### 📉 Matrice de confusion sur le test")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pas de complication", "Complication"])
    disp.plot(ax=ax, cmap="Blues")
    st.pyplot(fig)
