# Disaster Risk Prediction Using Machine Learning

## Project Overview

This project contains three machine learning models developed for the Kaavalal disaster management project in Kerala, India.

The models estimate numerical disaster risk scores using available environmental, historical, and susceptibility-related data.

## Models

### 1. Flood Risk Prediction
- Algorithm: Decision Tree Regressor
- Target: Flood susceptibility score
- Output: Predicted flood susceptibility score

### 2. Landslide Risk Prediction
- Algorithm: See the model training script
- Target: Landslide risk score
- Output: Predicted landslide risk score

### 3. Drought Risk Prediction
- Algorithm: Random Forest Regressor
- Target: Drought risk score
- Output: Predicted drought risk score

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## Project Structure

- `flood_risk_prediction.py` — Flood model training and evaluation
- `landslide_risk_prediction.py` — Landslide model training and evaluation
- `drought_risk_prediction.py` — Drought model training and evaluation

Dataset and output filenames may differ depending on the files included in the repository.

## How to Run

1. Install Python.
2. Install the required libraries.
3. Place the required dataset CSV files in the project folder.
4. Run the desired Python script.

Example:

```bash
python drought_risk_prediction.py
```

## Outputs

The scripts can generate model evaluation metrics, actual-versus-predicted CSV files, prediction graphs, and saved trained models.

## Important Note

This is an academic project. The predicted values are risk scores based on the available datasets and modelling approach. They should not be interpreted as validated real-world disaster probabilities without further validation.

## Author

Developed as part of the Kaavalal disaster management project.
