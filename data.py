import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_excel("AI_CHIR_DIG_FINAL.xlsx", sheet_name="Second cleanup")
    return df