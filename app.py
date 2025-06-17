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

st.title("Prédiction Complications Chirurgicales")

@st.cache_data
def load_data():
    df = pd.read_excel("AI_CHIR_DIG_FINAL.xlsx", sheet_name="Second cleanup", skiprows=2)
    df = df.replace('-', np.nan)
    df = df.dropna(subset=['COMPLICATION'])
    df = df.reset_index(drop=True)
    return df

df = load_data()

# Suppression des deux premières colonnes
df = df.iloc[:, 2:]  # à partir de la 3ème colonne

# Définir inputs (colonnes 3 à 41) et outputs (42 à 74)
input_cols = df.columns[0:39].tolist()  # index 0 à 38 (correspond à colonne 3 à 41 originales)
output_cols = df.columns[39:73].tolist()  # index 39 à 72 (correspond à colonne 42 à 74 originales)

st.markdown("### Colonnes d'entrée (inputs) détectées :")
st.write(input_cols)

st.markdown("### Colonnes de sortie possibles (outputs) :")
st.write(output_cols)

# Sélection dynamique des colonnes d'entrée (features)
input_cols = st.multiselect(
    "Sélectionnez les colonnes d'entrée (features) :",
    options=input_cols,
    default=input_cols  # on sélectionne par défaut toutes les colonnes inputs
)

# Sélection de la colonne cible parmi outputs
target_col = st.selectbox("Sélectionnez la colonne cible (target) :", options=output_cols)

if len(input_cols) == 0:
    st.warning("Veuillez sélectionner au moins une colonne d'entrée.")
    st.stop()

# Nettoyage et conversion
df_clean = df[input_cols + [target_col]].copy()

for col in input_cols + [target_col]:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

df_clean = df_clean.dropna(subset=input_cols + [target_col]).reset_index(drop=True)

X = df_clean[input_cols]
y = df_clean[target_col].astype(int)

# SMOTE pour rééquilibrage
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

# Trouver le meilleur modèle selon F1 macro
results.sort(key=lambda x: x[1], reverse=True)
best_model_name = results[0][0]
best_pipeline = pipelines[best_model_name]

st.markdown(f"### Meilleur modèle sélectionné : **{best_model_name}** (F1 score macro = {results[0][1]:.2f})")

# Saisie utilisateur dynamique
st.markdown("### Entrez les valeurs pour la prédiction")

user_input = {}
for col in input_cols:
    col_min = float(df_clean[col].min())
    col_max = float(df_clean[col].max())
    col_median = float(df_clean[col].median())
    user_input[col] = st.number_input(
        label=col,
        min_value=col_min,
        max_value=col_max,
        value=col_median
    )

if st.button("Prédire"):
    input_df = pd.DataFrame([user_input])
    pred = best_pipeline.predict(input_df)[0]
    proba = None
    if hasattr(best_pipeline.named_steps['clf'], "predict_proba"):
        proba = best_pipeline.predict_proba(input_df)[0][1]

    if pred == 0:
        st.error("Prediction : Complication")
    else:
        st.success("Prediction : Pas de complication")

    if proba is not None:
        st.write(f"Probabilité de complication : {proba:.2f}")

    # Affichage matrice de confusion et rapport classification
    y_pred_test = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    st.write("### Matrice de confusion sur le test")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Pas de complication", "Complication"])
    disp.plot(ax=ax, cmap="Blues")
    st.pyplot(fig)

    st.write("### Rapport de classification sur le test")
    report = classification_report(y_test, y_pred_test, target_names=["Pas de complication", "Complication"], zero_division=0)
    st.text(report)
