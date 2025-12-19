import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, recall_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

st.title("Predictive Analysis of Colectomy-Related Complications")

@st.cache_data
def load_data():
    df = pd.read_excel("AI_CHIR_DIG_FINAL.xlsx", sheet_name="Second cleanup")
    return df

df_raw = load_data()

# Définir les plages
input_range = df_raw.columns [:28]  # colonnes 3 à 41 (index 2 à 40)
output_range = df_raw.columns[28:43] # colonnes 42 à 74 (index 41 à 73)

st.markdown("### 🎯 Select Criteria")

selected_inputs = st.multiselect("🧮 Input Criteria", input_range, default=[])
selected_target = st.selectbox("🏷️ Expected Prediction", options=[""] + list(output_range))

if selected_target == "" or len(selected_inputs) == 0:
    st.warning("Choose at least one criteria and one prediction")
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

st.markdown(f"### ✅ Best Model : **{best_model_name}** (Recall = {best_recall_per_class[1]:.2f})")

st.markdown("### 🧾 Select variables for complication prediction")

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

if st.button("📊 Predict"):
    input_df = pd.DataFrame([user_input])
    # Get predicted probabilities and classes
    proba_all = best_pipeline.predict_proba(input_df)[0]
    classes = best_pipeline.named_steps['clf'].classes_

    # Manually choose the class with the highest probability
    pred = classes[np.argmax(proba_all)]

    if hasattr(best_pipeline.named_steps['clf'], "predict_proba"):
        proba_all = best_pipeline.predict_proba(input_df)[0]
        classes = best_pipeline.named_steps['clf'].classes_
        proba_pairs = sorted(zip(classes, proba_all), key=lambda x: x[1], reverse=True)
        
        st.markdown(f"### 📊 Predicted probabilities for {selected_target}")
        st.write( "Interpretation: for this patient, here is the estimated probability for " 
                 "each possible value of the target column." )
        
        for cls, p in proba_pairs:
            st.write(f"- **{selected_target} = {cls}** : {p*100:.1f}%")
        st.markdown(f"👉 Prediction : **{selected_target} = {pred}**")
        sorted_idx = np.argsort(proba_all)[::-1]
        best_cls = classes[sorted_idx[0]]
        best_p = proba_all[sorted_idx[0]]
        second_cls = classes[sorted_idx[1]]
        second_p = proba_all[sorted_idx[1]]
        st.markdown("### 🧾 Interpretation")
        st.write( f"For this patient, the model estimates that the probability of belonging to the category "
                 f"**{best_cls} ({best_p*100:.1f}%)** is higher than that of belonging "
                 f"to the category **{second_cls} ({second_p*100:.1f}%)**. "
                 f"👉 The model therefore chooses **{selected_target} = {best_cls}**." )
    # Matrice de confusion
    y_pred_test = best_pipeline.predict(X_test)
    classes_test = best_pipeline.named_steps['clf'].classes_
    cm = confusion_matrix(y_test, y_pred_test, labels=classes_test)

    st.markdown("### 📉 Confusion matrix on ({selected_target})")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Complication", "Complication"])
    disp.plot(ax=ax, cmap="Blues")
    st.pyplot(fig)

    # Analyse textuelle de la matrice de confusion
    st.markdown("### 🔎 Analysis of the confusion matrix")
    cm_values = cm.tolist()
    class_labels = list(classes_test)
    if len(class_labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        st.write(f"- **True negatives (TN)** : {tn} → {selected_target} = {class_labels[0]} correctly identified")
        st.write(f"- **False positives (FP)** : {fp} → {selected_target} = {class_labels[0]} but predicted as {class_labels[1]}")
        st.write(f"- **False negatives (FN)** : {fn} → {selected_target} = {class_labels[1]} but predicted as {class_labels[0]}")
        st.write(f"- **True positives (TP)** : {tp} → {selected_target} = {class_labels[1]} correctly identified")
        labels = sorted(class_labels)
        acc = accuracy_score(y_test, y_pred_test)
        report = classification_report(y_test, y_pred_test, labels=labels, target_names=[str(c) for c in class_labels], output_dict=True)
        
        if any(report[str(c)]["precision"] == 0 and report[str(c)]["recall"] == 0 for c in class_labels):
            st.info(
                "ℹ️ For some categories, the model did not predict any cases in the test set. "
                "As a result, precision and recall are shown as 0. "
                "This usually reflects class imbalance or limited sample size."
            )
    
        st.write(f"\n📊 **Overall accuracy** : {acc:.2f}")
        st.markdown("### 📐 Metrics by class")
        for cls in class_labels:
            precision = report[str(cls)]["precision"]
            recall = report[str(cls)]["recall"]
            f1 = report[str(cls)]["f1-score"]
            st.write(f"- **Class {cls}** → Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        with st.expander("ℹ️ Understanding the metrics (Precision, Recall, F1)"):
                
            st.markdown("""
        ### 📊 Understanding the Metrics

        - **Precision**  
        When the model says **"Complication"**, how often is it correct?  
        👉 The higher the precision, the fewer false alarms there are.

        - **Recall**  
        Out of all patients who will actually have a complication, how many does the model detect?  
        👉 The higher the recall, the fewer real cases are missed.

        - **F1-score**  
        A score that combines precision and recall.  
        👉 It balances avoiding too many false alarms and not missing real cases.

        ---

        🏥 **Simple Example**

        - The model predicts **"complication"** for 10 patients  
        - 7 actually have one → **Precision = 70%**  
        - Out of 20 patients who actually have a complication, it finds 12 → **Recall = 60%**  
        - The **F1-score = 64%**, which summarizes the trade-off
        """)
    else:
        # Cas multi-classes
        st.write("This target has several categories. Interpretation :")
        st.write("- The values on the diagonal represent the correctly predicted cases.")
        st.write("- The values off the diagonal represent classification errors.")
        st.write("👉 The darker the diagonal (higher value), the better the model.")
        st.write(f"Evaluated classes : {class_labels}")
