import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split

import config as cf
import data
import preprocessing
import models
import evaluate

st.warning(
    "⚠️ This tool is for research and educational purposes only. "
    "It must not be used as a sole basis for clinical decision-making."
)

st.title("Predictive Analysis of Colectomy-Related Complications")

df_raw = data.load_data()

# Définir les plages
input_range = df_raw.columns [:28]
output_range = df_raw.columns[28:43]

st.markdown("### 🎯 Select Criteria")

selected_inputs = st.multiselect(
    "🧮 Input Criteria",
    options=input_range,
    format_func=lambda x: cf.COLUMN_LABELS.get(x, x)
)

selected_target = st.selectbox(
    "🏷️ Expected Prediction",
    options=[""] + list(output_range),
    format_func=lambda x: cf.COLUMN_LABELS.get(x, x) if x != "" else ""
)

if selected_target == "" or len(selected_inputs) == 0:
    st.warning("Choose at least one criteria and one prediction")
    st.stop()

st.markdown("### ⚙️ Model optimization criterion")

metric_choice = st.selectbox(
    "Select the metric used to choose the best model",
    options=[
        "Recall (macro)",
        "Accuracy",
        "F1-score (macro)",
        "Precision (macro)"
    ]
)

selected_scoring = cf.SCORING_MAP[metric_choice]

df_model = df_raw.dropna(subset=selected_inputs + [selected_target]).reset_index(drop=True)

X = df_model[selected_inputs]
y = df_model[selected_target].values.ravel().astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

results = []
pipelines = {}

cat_cols, cont_cols, binary_cols = preprocessing.split_columns(X_train)

for name, config in models.MODEL_CONFIGS.items():
    print(f"Training {name}...")

    score, best_estimator = models.train_single_model(
        name=name,
        config=config,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        selected_scoring=selected_scoring,
        cat_cols=cat_cols,
        cont_cols=cont_cols,
        binary_cols=binary_cols
    )

    results.append((name, score, best_estimator))

    pipelines[name] = best_estimator

# Pick best
results.sort(key=lambda x: x[1], reverse=True)
best_model_name, best_score, best_pipeline = results[0]

st.markdown(
    f"""
    ### ✅ Selected Model

    **Model**: {best_model_name}  
    **Optimization metric**: {metric_choice}  
    **Score**: {best_score:.2f}

    ℹ️ The model was selected because it achieved the best performance according
    to the chosen metric.
    """
)

# =============================
# STEP 1 — Univariate evaluation
# =============================
st.markdown("## 🔬 Univariate model performance")
st.write(
    "Each variable is evaluated **alone** to measure its intrinsic predictive power. "
    "The final score is the **mean** over all variables."
)

univariate_scores = []

for feature in selected_inputs:
    X_single = df_model[[feature]]
    y_single = df_model[selected_target].astype(int)

    score = evaluate.evaluate_model_on_inputs(
        model=best_pipeline.named_steps["clf"],
        requires_encoding=models.MODEL_CONFIGS[best_model_name]["requires_encoding"],
        X=X_single,
        y=y_single,
        scoring=metric_choice
    )

    univariate_scores.append(score)
    st.write(f"• **{feature}** → {metric_choice}: `{score:.3f}`")

# =============================
# Mean univariate score
# =============================
mean_score = np.mean(univariate_scores)

st.success(
    f"📊 Mean {metric_choice} using **1 variable at a time**: `{mean_score:.3f}`"
)

st.markdown("### 🧪 Input count performance analysis")

with st.spinner("Evaluating combinations (this may take a while)..."):
    results = evaluate.evaluate_by_input_count(
        model=best_pipeline.named_steps["clf"],
        requires_encoding=models.MODEL_CONFIGS[best_model_name]["requires_encoding"],
        X=df_model[selected_inputs],
        y=y,
        scoring=metric_choice,
        max_inputs=1
    )

st.markdown("### 📊 Mean performance by number of inputs")

for k, res in results.items():
    st.write(
        f"**{k} input(s)** → "
        f"{metric_choice}: **{res['mean']:.3f} ± {res['std']:.3f}** "
        f"(n={res['n_combinations']})"
    )

st.markdown("### 🧾 Select variables for complication prediction")

user_input = {}

for col in selected_inputs:
    label = cf.COLUMN_LABELS.get(col, col)

    if pd.api.types.is_numeric_dtype(df_model[col]):
        user_input[col] = st.number_input(
            label,
            value=float(df_model[col].median())
        )
    else:
        user_input[col] = st.selectbox(
            label,
            options=df_model[col].dropna().unique().tolist()
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
        
        st.markdown(f"### 📊 Predicted probabilities for **{cf.COLUMN_LABELS.get(selected_target, selected_target)}**")
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

    st.markdown(f"### 📉 Confusion matrix on **{cf.COLUMN_LABELS.get(selected_target, selected_target)}**")
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
