
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# STEP 1: Load the flood dataset

data = pd.read_csv("flood_training.csv")

print("Flood dataset loaded successfully!")
print("Total rows:", len(data))
print("\nFirst five rows:")
print(data.head())


# STEP 2: Select the target column

target_column = "flood_susc_score_adj_0_100"


# STEP 3: Remove columns that should not be used as inputs

drop_columns = [
    target_column,
    "taluk_flood_score_0_100",
    "ac_no",
    "record_type",
    "verified",
    "severity_label",
    "severity_basis",
    "severity_confidence",
    "event_severity_0_3",
    "flood_susc_level",
    "modelled_exposure_0_100",
    "exposure_class",
    "elevation_basis",
    "taluk_mapping_basis",
    "constituency_inputs_matched",
    "history_record_matched"
]

X = data.drop(columns=drop_columns, errors="ignore")
y = pd.to_numeric(data[target_column], errors="coerce")


# STEP 4: Convert categorical input columns into numerical values

X = pd.get_dummies(X, dummy_na=True)

for column in X.columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")


# STEP 5: Remove invalid target rows and missing input columns

valid_rows = y.notna() & (y >= 0) & (y <= 100)

X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

X = X.dropna(axis=1, how="all")
X = X.fillna(X.median())

print("\nRows available for training:", len(y))
print("Number of input features:", len(X.columns))

if len(y) < 10:
    raise ValueError("Not enough valid rows for training.")


# STEP 6: Save the feature column names

feature_columns = X.columns.tolist()


# STEP 7: Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=1
)


# STEP 8: Create the Decision Tree model

model = DecisionTreeRegressor(
    max_depth=5,
    random_state=1
)


# STEP 9: Train the model

model.fit(X_train, y_train)

print("\nFlood risk model trained successfully!")


# STEP 10: Evaluate the model

train_predictions = model.predict(X_train)
y_pred = model.predict(X_test)

y_pred = y_pred.clip(0, 100)

train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print("\n----- MODEL EVALUATION -----")
print("Training MAE:", round(train_mae, 2))
print("Testing MAE:", round(test_mae, 2))
print("Testing R2 score:", round(test_r2, 3))


# STEP 11: Predict a new flood risk score
# Uses an existing dataset row as an example

new_data = X.iloc[[0]]

predicted_risk = model.predict(new_data)[0]
predicted_risk = max(0, min(100, predicted_risk))

print("\n----- NEW FLOOD RISK PREDICTION -----")
print("Predicted flood susceptibility score:", round(predicted_risk, 2))


# STEP 12: Display actual and predicted flood scores

results = pd.DataFrame({
    "Actual_Flood_Score": y_test.values,
    "Predicted_Flood_Score": y_pred.round(2)
})

print("\n----- ACTUAL VS PREDICTED FLOOD SCORES -----")
print(results.head(10).to_string(index=False))


# STEP 13: Save actual and predicted scores to CSV

results.to_csv(
    "kaavalal_flood_risk_predictions.csv",
    index=False
)

print("\nPrediction CSV saved successfully!")


# STEP 14: Plot actual vs predicted flood scores

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Flood Susceptibility Score")
plt.ylabel("Predicted Flood Susceptibility Score")
plt.title("Kaavalal Flood Risk Score Prediction")

plt.tight_layout()

plt.savefig(
    "kaavalal_flood_risk_prediction_plot.png",
    dpi=300
)

plt.show()

print("\nPrediction graph saved successfully!")


# STEP 15: Display feature importance

print("\n----- FEATURE IMPORTANCE -----")

importance_values = model.feature_importances_

for i in range(len(feature_columns)):
    print(
        feature_columns[i],
        ":",
        round(importance_values[i], 3)
    )


# STEP 16: Save the trained model and feature names

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns
    },
    "flood_risk_model.pkl"
)

print("\nModel saved as flood_risk_model.pkl")


# STEP 17: Confirm completion

print("\n----- ALL TASKS COMPLETED SUCCESSFULLY -----")
