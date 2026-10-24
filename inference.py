import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Load artifacts at startup. The files were dumped from
# notebook FC_Operations_Risk_Predictions.ipynb

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR/"model"

model_pipeline = joblib.load(MODEL_DIR/"random_forest_pipeline.pkl")
model_metadata = joblib.load(MODEL_DIR/"deployment_metadata.pkl")

class_threshold = model_metadata["classification_threshold"]
model_features = model_metadata["model_features"]

positive_class = model_metadata["positive_class"]
negative_class = model_metadata["negative_class"]

def calculate_capacity_gap(
    current_backlog,
    expected_incoming_work,
    packers_assigned,
    expected_packing_rate
):
    if current_backlog < 0 or expected_incoming_work < 0:
        raise ValueError("Work volumes cannot be negative.")

    if packers_assigned < 0:
        raise ValueError("Staffing cannot be negative.")
    
    if expected_packing_rate <= 0:
        raise ValueError("Packing rate cannot be 0 or negative.")

    available_work = (
        current_backlog + expected_incoming_work
    )

    line_capacity = (
        packers_assigned * expected_packing_rate
    )

    capacity_gap = max(0, available_work - line_capacity)

    return {
        "available_work": available_work,
        "line_capacity": line_capacity,
        "capacity_gap": capacity_gap
    }

def create_dataframe(capacity_gap):
    # Construct model input. Will be in the same format as what
    # was exported in the notebook FC_Operations_Risk_Predictions.ipynb
    input_data = pd.DataFrame([{"capacity_gap": capacity_gap}])

    # Enforce expected model features
    input_data = input_data[model_features]

    return input_data
    
def predict(
        current_backlog,
        expected_incoming_work,
        packers_assigned,
        expected_packing_rate
        ):

    features = calculate_capacity_gap(
        current_backlog,
        expected_incoming_work,
        packers_assigned,
        expected_packing_rate
    )

    X = create_dataframe(features["capacity_gap"])
    risk_score = float(model_pipeline.predict_proba(X)[0, 1])

    return {
        **features,
        "risk_score": risk_score,
        "high_risk": risk_score >= class_threshold
    }

