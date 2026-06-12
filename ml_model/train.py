import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv("../data/traffic.csv")

# Convert time
df["date_time"] = pd.to_datetime(df["date_time"])
df["hour"] = df["date_time"].dt.hour
df["day"] = df["date_time"].dt.dayofweek

# Encode holiday (None → 0, holiday → 1)
df["holiday"] = df["holiday"].apply(lambda x: 0 if x == "None" else 1)

# Drop unused text columns
df = df.drop(columns=["weather_main", "weather_description", "date_time"])

# Features
X = df.drop(columns=["traffic_volume"])
y = df["traffic_volume"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = XGBRegressor(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved successfully")