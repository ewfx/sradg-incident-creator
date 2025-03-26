import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from transformers import pipeline
from sklearn.preprocessing import LabelEncoder

# Load reconciliation dataset
df = pd.read_csv("daily_reconciliation_report.csv")

# Fill missing values
df.fillna("", inplace=True)

# Identify numerical and categorical columns dynamically
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Remove irrelevant text columns (assumption: "Break Comment" is text, not a feature)
irrelevant_cols = ["Break Comment", "Notes", "Description"]  # Adjust based on dataset
categorical_cols = [col for col in categorical_cols if col not in irrelevant_cols]

# Convert categorical features to numerical using Label Encoding
encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))  # Convert to string for encoding

# Convert date columns to numerical (days since epoch)
for col in df.select_dtypes(include=['datetime64']).columns:
    df[col] = (df[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')

# Fill NaN values after conversion
df[numerical_cols + categorical_cols] = df[numerical_cols + categorical_cols].fillna(0)

# Train Isolation Forest model dynamically
iso_forest = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly_Score'] = iso_forest.fit_predict(df[numerical_cols + categorical_cols])
df['Anomaly'] = df['Anomaly_Score'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

# Load LLM-based anomaly classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define anomaly categories
categories = ["Data Entry Error", "Missing Transaction", "Currency Mismatch", "Timing Issue"]

# Improved classification logic
def classify_anomaly(row):
    if row['Anomaly'] == 'Anomaly':
        comment = str(row.get('Break Comment', '')).lower()
        balance_diff = row.get('Balance Difference', 0)

        # Rule-based classification
        if "fx rate fluctuation" in comment or "currency" in comment:
            return "Currency Mismatch"
        elif "timing" in comment or "settlement delay" in comment:
            return "Timing Issue"
        elif "missing" in comment or balance_diff > 10000:  # Large difference may indicate missing txn
            return "Missing Transaction"

        # Use LLM for classification if no rule-based match
        prompt = f"Anomaly detected: Balance difference = {balance_diff}. Reason: {comment}"
        result = classifier(prompt, candidate_labels=categories)
        return result['labels'][0]

    return "No Issue"

df['Anomaly_Type'] = df.apply(classify_anomaly, axis=1)

df.to_csv("reconciliation_results_dynamic.csv", index=False)
print("Anomaly detection and classification completed with dynamic feature selection.")