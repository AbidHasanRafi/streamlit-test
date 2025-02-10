import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model

# Load the trained LSTM model
model = load_model("swell_model_lstm.h5")

# Define the feature names with user-friendly labels
feature_labels = {
    'MEAN_RR': "Mean Time Between Heartbeats",
    'MEDIAN_RR': "Median Time Between Heartbeats",
    'SDRR': "Standard Deviation of RR Intervals",
    'RMSSD': "Root Mean Square of Successive Differences",
    'SDSD': "Standard Deviation of Successive Differences",
    'SDRR_RMSSD': "Ratio of SDRR to RMSSD",
    'HR': "Heart Rate",
    'pNN25': "Percentage of RR Intervals > 25ms",
    'pNN50': "Percentage of RR Intervals > 50ms",
    'SD1': "Short-Term HRV Component",
    'SD2': "Long-Term HRV Component",
    'KURT': "Kurtosis of RR Intervals",
    'SKEW': "Skewness of RR Intervals",
    'MEAN_REL_RR': "Mean Relative RR Interval",
    'MEDIAN_REL_RR': "Median Relative RR Interval",
    'SDRR_REL_RR': "Standard Deviation of Relative RR",
    'RMSSD_REL_RR': "RMSSD of Relative RR",
    'SDSD_REL_RR': "SDSD of Relative RR",
    'SDRR_RMSSD_REL_RR': "Ratio of SDRR to RMSSD (Relative)",
    'KURT_REL_RR': "Kurtosis of Relative RR",
    'SKEW_REL_RR': "Skewness of Relative RR",
    'VLF': "Very Low-Frequency Power",
    'VLF_PCT': "Percentage of VLF Power",
    'LF': "Low-Frequency Power",
    'LF_PCT': "Percentage of LF Power",
    'LF_NU': "Normalized LF Power",
    'HF': "High-Frequency Power",
    'HF_PCT': "Percentage of HF Power",
    'HF_NU': "Normalized HF Power",
    'TP': "Total Power",
    'LF_HF': "LF/HF Ratio",
    'HF_LF': "HF/LF Ratio",
    'sampen': "Sample Entropy",
    'higuci': "Higuchi Fractal Dimension"
}

# Load LabelEncoder for class labels
label_encoder = LabelEncoder()
label_encoder.fit(["No Stress", "Time Pressure", "Interruption"])

# Streamlit UI
st.title("📊 Stress Prediction")
st.markdown(
    """
    This app predicts **workplace stress conditions** based on **heart rate variability (HRV)** data.
    """
)

# Button to fill fields with random values
if st.button("🎲 Fill Random Values for Testing"):
    random_values = {key: np.round(np.random.uniform(-50, 50), 2) for key in feature_labels}
else:
    random_values = {key: 0.0 for key in feature_labels}  # Default values

# Collect user inputs
inputs = {}
for key, label in feature_labels.items():
    inputs[key] = st.number_input(f"{label}:", min_value=-100.0, max_value=100.0, value=random_values[key], step=0.01)

# Predict button
if st.button("🔍 Predict Stress Level"):
    try:
        # Convert input data into NumPy array
        input_data = np.array(list(inputs.values())).reshape(1, 1, -1)

        # Standardize input data (using a fixed scaler)
        scaler = StandardScaler()
        scaler.fit(np.zeros((10, len(inputs))))  # Fit on dummy data to avoid shape errors
        input_data = scaler.transform(input_data[0].reshape(1, -1)).reshape(1, 1, -1)

        # Get model prediction
        prediction = model.predict(input_data)

        # Ensure the prediction is valid
        if prediction.size == 0:
            st.error("⚠️ Error: Model did not return a valid prediction.")
        else:
            # Decode prediction
            predicted_condition = label_encoder.inverse_transform([np.argmax(prediction)])
            st.success(f"✅ Predicted Stress Condition: **{predicted_condition[0]}**")
    
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
