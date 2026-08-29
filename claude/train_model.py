"""
Train the Seoul Bike Demand model and save it as a pickle file
for the Streamlit app to load.

Run this once locally (with SeoulBikeData.csv in the same folder):
    python train_model.py

It will create: model.pkl
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1. Load dataset
# -----------------------------
# NOTE: SeoulBikeData.csv often uses latin-1/cp1252 encoding because of the
# "°C" characters, so encoding is set explicitly to avoid read errors.
df = pd.read_csv("SeoulBikeData.csv", encoding="latin-1")

# -----------------------------
# 2. Feature engineering (same as notebook)
# -----------------------------
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values(["Date", "Hour"]).reset_index(drop=True)

df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

target = "Rented Bike Count"

numerical_features = [
    "Hour",
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
    "DayOfWeek",
    "Month",
    "IsWeekend",
]

categorical_features = ["Seasons", "Holiday", "Functioning Day"]

features = numerical_features + categorical_features

X = df[features]
y = df[target]

# -----------------------------
# 3. Chronological 80/20 split
# -----------------------------
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# -----------------------------
# 4. Preprocessing + model pipeline (Gradient Boosting — best model)
# -----------------------------
gb_preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

gb_model = Pipeline(
    steps=[
        ("preprocessor", gb_preprocessor),
        ("model", GradientBoostingRegressor(random_state=42)),
    ]
)

gb_param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3, 5],
    "model__min_samples_leaf": [1, 2, 5],
}

gb_grid = GridSearchCV(
    gb_model,
    gb_param_grid,
    cv=TimeSeriesSplit(n_splits=5),
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)

print("Training Gradient Boosting model (this can take a few minutes)...")
gb_grid.fit(X_train, y_train)

gb_final = gb_grid.best_estimator_
gb_final.fit(X_train, y_train)

print("Best params:", gb_grid.best_params_)

# -----------------------------
# 5. Evaluate
# -----------------------------
test_pred = gb_final.predict(X_test)
print("Test MAE:", mean_absolute_error(y_test, test_pred))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, test_pred)))
print("Test R2:", r2_score(y_test, test_pred))

# -----------------------------
# 6. Save the trained pipeline + metadata the app needs
# -----------------------------
artifact = {
    "model": gb_final,
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "features": features,
    "categorical_options": {
        col: sorted(df[col].unique().tolist()) for col in categorical_features
    },
}

with open("model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\nSaved trained pipeline to model.pkl")
