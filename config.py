# Human-readable labels for display
COLUMN_LABELS = {
    # ===== INPUTS =====
    "Age_surgery": "Age",
    "Sex": "Sex",
    "FHX": "Risk factors",
    "FHX_1DEGREE": "Family factors (1st degree)",
    "TOBACCO": "Tobacco use",
    "BMI": "Body Mass Index (BMI)",
    "PHX_PULM": "History of lung disease",
    "PHX_CVD": "History of cardiovascular disease",
    "ATCD_COLONO": "Previous colonoscopy",
    "ATCD_COLONO_AGE": "Age at last colonoscopy",
    "PHX_DM": "History of diabetes mellitus",
    "CONTEXT_NO_SP": "Asymptomatic",
    "CONTEXT_BLOODY_STOOLS_ANEMIA": "Bloody stools / Anemia",
    "CONTEXT_OCCLUSION": "Occlusion",
    "CONTEXT_PERFORATION": "Perforation",
    "CONTEXT_PAIN_DISCOMFORT": "Abdominal pain / discomfort",
    "CONTEXT_AEG_FEVER": "Fever / fatigue",
    "CONTEXT_TRANSIT_DISORDER": "Transit disorder",
    "CI_PREOP": "Preoperative contraindication",
    "PERFORATIO_PRE_OP": "Preoperative perforation",
    "ABCESS_PRE_OP": "Preoperative abscess",
    "PERITONITIS_PRE_OP": "Preoperative peritonitis",
    "METASTATIC_CANCER_PRE_OP": "Metastatic cancer (preop)",
    "ASA": "ASA score",
    "HB_PREOP": "Preoperative hemoglobin",
    "OPERATION_AVANT/APRES_MIDI": "Surgery time (AM / PM)",
    "pT": "Clinical T stage",
    "N0_N+": "Clinical N stage",

    # ===== OUTPUTS =====
    "CONVERSION": "Conversion",
    "PEROP_BLEEDING": "Peroperative bleeding",
    "INSTABILITY": "Hemodynamic instability (perop)",
    "ICU_ADMISSION": "ICU admission (postop)",
    "REINTERVENTION": "Reintervention",
    "COMPLICATION": "Complication (any)",
    "COMPLICATION_MAJOR": "Major complication",
    "COMPLICATION_INFX": "Postoperative infection",
    "COMPLICATION_WOUND_INFX": "Postoperative wound infection",
    "COMPLICATION_LEAKAGE": "Postoperative anastomosis leakage",
    "COMPLICATION_ABCESS": "Postoperative abscess",
    "COMPLICATION_ILEUS_NO1_YES2_NO_GAS": "Postoperative ileus",
    "COMPLICATION_BLEED": "Postoperative bleeding",
    "COMPLICATION_GRADE.DINDO": "Postoperative cancer upstaging"
}

SCORING_MAP = {
    "Recall (macro)": "recall_macro",
    "Accuracy": "accuracy",
    "F1-score (macro)": "f1_macro",
    "Precision (macro)": "precision_macro"
}