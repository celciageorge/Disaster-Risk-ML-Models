import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# STEP 1: Load the corrected drought dataset

data = pd.read_csv("drought_training.csv")

print("Corrected drought dataset loaded successfully!")
print("Total rows:", len(data))
print("\nFirst five rows:")
print(data.head())


# STEP 2: Select the input features and target column

feature_columns = [
    "rainfall_deficit_pct",
    "temp_anomaly_c",
    "drought_susceptibility_score",
    "historical_drought_events"
]

target_column = "drought_risk_pct"


# STEP 3: Check whether all required columns exist

required_columns = feature_columns + [target_column]

for column in required_columns:
    if column not in data.columns:
        raise ValueError("Missing required column: " + column)


# STEP 4: Convert the required column values into numbers

for column in required_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


# STEP 5: Remove rows containing missing values in required columns
# Keep only rows where the drought risk score is between 0 and 100

data = data.dropna(subset=required_columns)

data = data[
    (data[target_column] >= 0)
    & (data[target_column] <= 100)
]

print("\nRows available for training:", len(data))

if len(data) < 10:
    raise ValueError("Not enough valid rows for training.")


# STEP 6: Separate input features (X) and target values (y)

X = data[feature_columns]
y = data[target_column]


# STEP 7: Split the dataset into training and testing data
# Use 80% for training and 20% for testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# STEP 8: Create the Random Forest regression model

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    min_samples_leaf=3
)


# STEP 9: Train the model using the training data

model.fit(X_train, y_train)

print("\nDrought risk regression model trained successfully!")


# STEP 10: Evaluate the model using training and testing data
# MAE measures the average prediction error
# R2 measures how well the model fits the test data

train_predictions = model.predict(X_train)
y_pred = model.predict(X_test)

train_mae = mean_absolute_error(
    y_train,
    train_predictions
)

test_mae = mean_absolute_error(
    y_test,
    y_pred
)

test_r2 = r2_score(
    y_test,
    y_pred
)

print("\n----- MODEL EVALUATION -----")
print("Training MAE:", round(train_mae, 2))
print("Testing MAE:", round(test_mae, 2))
print("Testing R2 score:", round(test_r2, 3))


# STEP 11: Predict the drought risk score for a new observation
# These are example values, not live weather data

new_data = pd.DataFrame({
    "rainfall_deficit_pct": [0.5],
    "temp_anomaly_c": [-0.03],
    "drought_susceptibility_score": [15],
    "historical_drought_events": [1]
})

predicted_risk = model.predict(new_data)[0]

# Keep the predicted score within the range of 0 to 100

predicted_risk = max(0, min(100, predicted_risk))

print("\n----- NEW DROUGHT RISK PREDICTION -----")
print("Predicted drought risk score:", round(predicted_risk, 2), "%")


# STEP 12: Display actual and predicted drought risk scores
# Compare the actual scores with the model's predictions on the test data

results = pd.DataFrame({
    "Actual_Drought_Score": y_test.values,
    "Predicted_Drought_Score": y_pred.round(2)
})

print("\n----- ACTUAL VS PREDICTED DROUGHT SCORES -----")
print(results.head(10).to_string(index=False))


# STEP 13: Save actual and predicted scores to a CSV file
# The CSV file will be created in the current working directory

results.to_csv(
    "kaavalal_drought_risk_predictions.csv",
    index=False
)

print("\nPrediction results saved successfully!")
print("File: kaavalal_drought_risk_predictions.csv")


# STEP 14: Plot actual drought scores against predicted scores
# Each point represents one test observation

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Drought Risk Score")
plt.ylabel("Predicted Drought Risk Score")
plt.title("Kaavalal Drought Risk Score Prediction")

plt.tight_layout()

# Save the graph as an image file

plt.savefig(
    "kaavalal_drought_risk_prediction_plot.png",
    dpi=300
)

# Display the graph on the screen

plt.show()

print("\nPrediction graph saved successfully!")
print("File: kaavalal_drought_risk_prediction_plot.png")


# STEP 15: Display the importance of each input feature
# Higher importance means the model relied more on that feature
# It does not necessarily mean that the feature causes drought

print("\n----- FEATURE IMPORTANCE -----")

importance_values = model.feature_importances_

for i in range(len(feature_columns)):
    print(
        feature_columns[i],
        ":",
        round(importance_values[i], 3)
    )


# STEP 16: Save the trained model and feature-column names
# These can be loaded later to make predictions without retraining

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns
    },
    "drought_risk_model.pkl"
)

print("\nModel saved as drought_risk_model.pkl")


# STEP 17: Confirm that all tasks are completed

print("\n----- ALL TASKS COMPLETED SUCCESSFULLY -----")