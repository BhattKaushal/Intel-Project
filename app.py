import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model, expected_columns = joblib.load("improved_model.pkl")

st.set_page_config(page_title="Smart Job Delay Predictor", layout="wide")
st.markdown("<h1 style='text-align:center;'>🚛 Smart Job Delay Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Upload your job file and instantly predict delays using an AI-powered model</p>", unsafe_allow_html=True)

# Sidebar for threshold
threshold = st.sidebar.slider("🎯 Confidence Threshold for Delay", 0.0, 1.0, 0.7, 0.01)
st.sidebar.markdown("Only predict ❌ delay if confidence ≥ selected threshold.")

# Preprocess function
def preprocess_input(df):
    df = df.copy()
    df["Dependencies"] = df["Dependencies"].apply(lambda x: 0 if pd.isna(x) or str(x).strip().lower() in ["", "none"] else 1)
    df["Resources_Available"] = df["Resources_Available"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["Estimated_End_Time"] = pd.to_datetime(df["Estimated_End_Time"], errors="coerce")
    df["Start_Day"] = df["Start_Time"].dt.dayofweek
    df["ETA_Day"] = df["Estimated_End_Time"].dt.dayofweek
    df["Time_to_Complete_hr"] = (df["Estimated_End_Time"] - df["Start_Time"]).dt.total_seconds() / 3600.0
    df["Critical"] = (df["Time_to_Complete_hr"] < 3).astype(int)
    df.drop(columns=["Start_Time", "Estimated_End_Time"], inplace=True)
    df = pd.get_dummies(df, columns=["Job_Status", "Location"])
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    return df[expected_columns]

# File uploader
uploaded_file = st.file_uploader("📂 Upload a CSV file with job data", type=["csv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data (first 10 rows)")
    st.dataframe(raw_df.head(10), use_container_width=True)

    try:
        processed_df = preprocess_input(raw_df)
        prob = model.predict_proba(processed_df)[:, 1]
        preds = (prob >= threshold).astype(int)

        result_df = raw_df.copy()
        result_df["Predicted_Status"] = np.where(preds == 1, "❌ Delayed", "✅ On-Time")
        result_df["Confidence (%)"] = (prob * 100).round(1)

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🧾 Total Jobs", len(result_df))
        with col2:
            st.metric("✅ On-Time", (preds == 0).sum())
        with col3:
            st.metric("❌ Predicted Delayed", (preds == 1).sum())

        st.subheader("📊 Prediction Results")
        st.dataframe(result_df[["Job_ID", "Location", "Job_Status", "Predicted_Status", "Confidence (%)"]], use_container_width=True)

        with st.expander("⬇ Download Results"):
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, file_name="predictions.csv", mime="text/csv")

        with st.expander("📌 Show only predicted delays"):
            st.dataframe(result_df[result_df["Predicted_Status"] == "❌ Delayed"], use_container_width=True)

    except Exception as e:
        st.error(f"Error while processing: {e}")
