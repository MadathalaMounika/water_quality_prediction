import os
import joblib
import streamlit as st

@st.cache_resource
def load_model():

    model_path = "water_quality_bundle.joblib"

    if not os.path.exists(model_path):
        st.error(
            f"Model file not found: {model_path}"
        )
        st.info(
            "Please upload your trained model file "
            "to your GitHub repository."
        )
        st.stop()

    bundle = joblib.load(model_path)

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    return model, feature_columns

model, feature_columns = load_model()
