import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("Ev_blast.pkl")
encoder = joblib.load("label_encoder.pkl")

st.title("Battery Health Prediction")

# Input fields
Battery_type = st.selectbox(
    "Battery Type",
    encoder["Battery type"].classes_
)

Poor_Cell_Design = st.number_input(
    "Poor Cell Design",
    min_value=0,
    max_value=1,
    step=1
)

External_Abuse = st.selectbox(
    "External Abuse",
    encoder["External Abuse"].classes_
)

Poor_Battery_Design = st.number_input(
    "Poor Battery Design",
    min_value=0,
    max_value=1,
    step=1
)

Short_Circuits = st.number_input(
    "Short Circuits",
    min_value=0,
    max_value=1,
    step=1
)

Temperature = st.number_input(
    "Temperature"
)

Overcharge_Overdischarge = st.selectbox(
    "Overcharge Overdischarge",
    encoder["Overcharge Overdischarge"].classes_
)

Battery_Maintenance = st.selectbox(
    "Battery Maintenance",
    encoder["Battery Maintenance"].classes_
)

# Create DataFrame
df = pd.DataFrame({
    "Battery type": [Battery_type],
    "Poor Cell Design": [Poor_Cell_Design],
    "External Abuse": [External_Abuse],
    "Poor Battery Design": [Poor_Battery_Design],
    "Short Circuits": [Short_Circuits],
    "Temperature": [Temperature],
    "Overcharge Overdischarge": [Overcharge_Overdischarge],
    "Battery Maintenance": [Battery_Maintenance]
})

# Prediction
if st.button("Predict Battery Health"):

    # Encode categorical columns
    for col in encoder:
        if col in df.columns:
            df[col] = encoder[col].transform(df[col])

    prediction = model.predict(df)[0]

    if prediction == 0:
        st.info("Moderate")
    elif prediction == 1:
        st.warning("Chance of Blast")
    elif prediction == 2:
        st.error("Blast")
    elif prediction == 3:
        st.success("Good")
    else:
        st.success(f"Predicted Class: {prediction}")
