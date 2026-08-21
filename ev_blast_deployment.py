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
    encoder["Battery_Type"].classes_
)

Poor_Cell_Design = st.number_input(
    "Poor Cell Design",
    min_value=0,
    max_value=1,
    step=1
)

External_Abuse = st.selectbox(
    "External Abuse",
    encoder["External_Abuse"].classes_
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
    encoder["Overcharge_Overdischarge"].classes_
)

Battery_Maintenance = st.selectbox(
    "Battery Maintenance",
    encoder["Battery_Maintenance"].classes_
)

# Create DataFrame
df = pd.DataFrame({
    "Battery_Type": [Battery_type],
    "Poor_Cell_Design": [Poor_Cell_Design],
    "External_Abuse": [External_Abuse],
    "Poor_Battery_Design": [Poor_Battery_Design],
    "Short_Circuits": [Short_Circuits],
    "Temperature": [Temperature],
    "Overcharge_Overdischarge": [Overcharge_Overdischarge],
    "Battery_Maintenance": [Battery_Maintenance]
})

# Prediction
if st.button("Predict Battery Health"):

    # Encode categorical columns
    categorical_columns = [
        "Battery_Type",
        "External_Abuse",
        "Overcharge_Overdischarge",
        "Battery_Maintenance"
    ]

    for col in categorical_columns:
        df[col] = encoder[col].transform(df[col])

    # Make prediction
    prediction = model.predict(df)[0]

    # Decode prediction
    prediction_label = encoder["Battery_Health"].inverse_transform([prediction])[0]

    # Display result
    if prediction_label == "Moderate":
        st.info("Moderate")

    elif prediction_label == "Chance_of_Blast":
        st.warning("Chance of Blast")

    elif prediction_label == "Blast":
        st.error("Blast")

    elif prediction_label == "Good":
        st.success("Good")

    else:
        st.success(f"Predicted Class: {prediction_label}")
