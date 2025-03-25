import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from io import BytesIO


def detect_anomalies(df):
    """Applies Isolation Forest to detect anomalies."""
    model = IsolationForest(contamination=0.05, random_state=42)
    df = df.select_dtypes(include=[np.number])  # Use only numeric columns
    df['anomaly'] = model.fit_predict(df)
    df['anomaly'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})
    return df


def convert_df_to_csv(df):
    """Converts a DataFrame to CSV format for download."""
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()


# Streamlit UI
st.title("🔍 File Reconciliation & Anomaly Detection")

# File Upload
uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data:", df.head())

    # Perform reconciliation & anomaly detection
    df_result = detect_anomalies(df)
    st.write("### Reconciled Data with Anomalies:")
    st.dataframe(df_result)

    # Download option
    csv_data = convert_df_to_csv(df_result)
    st.download_button(
        label="📥 Download Results",
        data=csv_data,
        file_name="reconciled_data.csv",
        mime="text/csv"
    )
