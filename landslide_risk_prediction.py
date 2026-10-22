
# Kaavalal Landslide Risk Score Prediction
# Predict a numerical landslide risk score from 0 to 100
# Uses all available simulated and documented records

# STEP 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# STEP 2: Load the dataset
df = pd.read_csv("landslide_training_data.csv")

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# Display real and simulated record counts
if "label_source" in df.columns:
    print("\nLabel source counts:")
    print(df["label_source"].value_counts(dropna=False))

print("\nTotal records loaded:", len(df))


# STEP 3: Check required columns
target = "risk_score_pct"

required_columns = [
    "constituency",
    "district",
    "year",
    "rainfall_72h_mm",
    "rainfall_7d_mm",
    "rainfall_30d_mm",
    "slope_deg",
    "elevation_m",
    "landslide_susceptibility",
    target
]

missing_columns = []

for column in required_columns:
    if column not in df.columns:
        missing_columns.append(column)

if len(missing_columns) > 0:
    raise ValueError(
        "Missing required columns: " + str(missing_columns)
    )


# STEP 4: Convert numerical columns to numeric values
numeric_columns = [
    "year",
    "rainfall_72h_mm",
    "rainfall_7d_mm",
    "rainfall_30d_mm",
    "slope_deg",
    "elevation_m",
    "landslide_susceptibility",
    target
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# STEP 5: Remove rows with missing target values
# Keep both simulated and documented records
df = df.dropna(subset=[target]).copy()

# Keep risk scores within the expected 0-100 range
df = df[
    (df[target] >= 0) &
    (df[target] <= 100)
].copy()

print("\nRecords available for training:", len(df))

if len(df) < 5:
    raise ValueError(
        "Not enough valid risk-score records to train the model."
    )


# STEP 6: Select input features (X)
feature_columns = [
    "constituency",
    "district",
    "year",
    "rainfall_72h_mm",
    "rainfall_7d_mm",
    "rainfall_30d_mm",
    "slope_deg",
    "elevation_m",
    "landslide_susceptibility"
]

X = df[feature_columns].copy()

# Select target column (Y)
y = df[target].copy()


# STEP 7: Convert text columns into numerical columns
X = pd.get_dummies(
    X,
    columns=["constituency", "district"],
    dummy_na=True,
    dtype=int
)

# Fill missing input values with their column medians
X = X.fillna(X.median(numeric_only=True))

# Remove columns containing only missing values
X = X.dropna(axis=1, how="all")

# Store the final input column names
model_feature_columns = X.columns.tolist()

print("\nInput features after encoding:")
print(model_feature_columns)

print("\nPrepared dataset shape:", X.shape)


# STEP 8: Split data into training and testing sets
# 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# STEP 9: Create the Random Forest Regression model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)


# STEP 10: Train the model
model.fit(X_train, y_train)

print("\nLandslide regression model trained successfully!")


# STEP 11: Predict risk scores for test records
y_pred = model.predict(X_test)

# Keep predictions within 0-100
y_pred = np.clip(y_pred, 0, 100)


# STEP 12: Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------")
print("Mean Absolute Error:", round(mae, 3))
print("Root Mean Squared Error:", round(rmse, 3))
print("R2 Score:", round(r2, 3))


# STEP 13: Display actual and predicted scores
results = pd.DataFrame({
    "Actual_Risk_Score": y_test.values,
    "Predicted_Risk_Score": y_pred.round(2)
})

print("\nActual vs Predicted Risk Scores:")
print(results.head(10).to_string(index=False))

results.to_csv(
    "kaavalal_landslide_risk_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")


# STEP 14: Plot actual vs predicted risk scores
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Landslide Risk Score")
plt.ylabel("Predicted Landslide Risk Score")
plt.title("Kaavalal Landslide Risk Score Prediction")

plt.tight_layout()

plt.savefig(
    "kaavalal_landslide_risk_prediction_plot.png"
)

plt.show()


# STEP 15: Compare training and testing performance
train_pred = np.clip(
    model.predict(X_train),
    0,
    100
)

test_pred = np.clip(
    model.predict(X_test),
    0,
    100
)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print("\nTraining MAE:", round(train_mae, 3))
print("Testing MAE:", round(test_mae, 3))

print("Training R2:", round(train_r2, 3))
print("Testing R2:", round(test_r2, 3))


# STEP 16: Check risk-score distribution
print("\nRisk Score Statistics:")
print(y.describe())

plt.figure(figsize=(8, 5))

plt.hist(
    y,
    bins=20,
    edgecolor="black"
)

plt.xlabel("Landslide Risk Score")
plt.ylabel("Number of Records")
plt.title("Distribution of Landslide Risk Scores")

plt.tight_layout()

plt.savefig(
    "kaavalal_landslide_risk_distribution.png"
)

plt.show()


# STEP 17: Save the trained model using joblib
# Store the model and the input column names together
model_data = {
    "model": model,
    "feature_columns": model_feature_columns,
    "input_columns": feature_columns,
    "target_column": target
}

joblib.dump(
    model_data,
    "kaavalal_landslide_risk_model.pkl"
)

print("\nTrained model saved successfully!")
print("File: kaavalal_landslide_risk_model.pkl")
