import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv("diseasepredictionfile.csv.csv", encoding="latin1")

# Drop NaN
df.dropna(inplace=True)

# Encode input features
feature_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    if col != "Disease Name":  # Skip target column
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        feature_encoders[col] = le

# Encode target labels
target_encoder = LabelEncoder()
df["Disease Name"] = target_encoder.fit_transform(df["Disease Name"])

# Split data
X = df.drop(["Disease Name", "Patient ID"], axis=1)
Y = df["Disease Name"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Save model + both encoders
with open("diseasepredictjsonfile.pkl", "wb") as f:
    pickle.dump((model, feature_encoders, target_encoder), f)
