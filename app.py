
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Water Quality Prediction",
    page_icon="💧",
    layout="wide"
)

@st.cache_resource
def load_model():
    bundle = joblib.load("water_quality_bundle.joblib")
    return bundle["model"], bundle["feature_columns"]

model, feature_columns = load_model()

st.title("💧 Water Quality Prediction System")
st.markdown(
    "Machine Learning Based Water Potability Prediction"
)

st.sidebar.header("Water Quality Parameters")

inputs = {}

for feature in feature_columns:
    inputs[feature] = st.sidebar.number_input(
        feature,
        value=0.0
    )

if st.button("Predict Water Quality", type="primary"):
    input_df = pd.DataFrame(
        [inputs],
        columns=feature_columns
    )

    prediction = int(model.predict(input_df)[0])

    if prediction == 1:
        st.success("💧 Prediction: Potable Water")
    else:
        st.error("⚠️ Prediction: Not Potable Water")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        if 1 in classes:
            potable_probability = probabilities[classes.index(1)]

            st.metric(
                "Potability Probability",
                f"{potable_probability * 100:.2f}%"
            )

st.divider()

st.subheader("📁 Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload a water quality CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    missing_columns = [
        col for col in feature_columns
        if col not in data.columns
    ]

    if missing_columns:
        st.error(
            f"Missing columns: {missing_columns}"
        )
    else:
        predictions = model.predict(
            data[feature_columns]
        )

        data["Predicted_Potability"] = predictions

        data["Prediction_Label"] = data[
            "Predicted_Potability"
        ].map({
            0: "Not Potable",
            1: "Potable"
        })

        st.dataframe(data, use_container_width=True)

        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Predictions",
            data=csv,
            file_name="water_quality_predictions.csv",
            mime="text/csv"
        )

st.caption(
    "⚠️ Educational use only. This prediction does not replace "
    "certified laboratory water testing."
)
