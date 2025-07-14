# Install this if you haven't already:
# pip install xgboost imbalanced-learn scikit-learn pandas joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load your original dataset
df = pd.read_csv("data.csv")

# Cleaning & Encoding
df["Dependencies"] = df["Dependencies"].apply(lambda x: 0 if pd.isna(x) or str(x).strip().lower() in ["", "none"] else 1)
df["Resources_Available"] = df["Resources_Available"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
df["Estimated_End_Time"] = pd.to_datetime(df["Estimated_End_Time"], errors="coerce")

# Feature Engineering
df["Start_Day"] = df["Start_Time"].dt.dayofweek
df["ETA_Day"] = df["Estimated_End_Time"].dt.dayofweek
df["Time_to_Complete_hr"] = (df["Estimated_End_Time"] - df["Start_Time"]).dt.total_seconds() / 3600.0
df["Critical"] = (df["Time_to_Complete_hr"] < 3).astype(int)

# Select features
X = df.drop(columns=["Job_ID", "Order_ID", "Actual_End_Time", "Delay_Reason", "Start_Time", "Estimated_End_Time", "Delay_Flag"])
y = df["Delay_Flag"]

# One-hot encode
X_encoded = pd.get_dummies(X, columns=["Job_Status", "Location"])

# Save full column list for app use
all_columns = X_encoded.columns.tolist()

# Oversample minority class
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_encoded, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model + feature list
joblib.dump((model, all_columns), "improved_model.pkl")
print("\n✅ Model saved to improved_model.pkl")
