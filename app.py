import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Water Quality Prediction",
    page_icon="💧",
    layout="wide"
)

@st.cache_resource
def load_model():

    model_path = "water_quality_model.pkl"

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)

    return model

model = load_model()
