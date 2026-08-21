import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Load artifacts at startup. The files were dumped from
# notebook FC_Operations_Risk_Predictions.ipynb

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR/"model"

model_pipeline = joblib.load(MODEL_DIR/"logistic_backlog_pipeline.pkl")
model_metadata = joblib.load(MODEL_DIR/"deployment_metadata.pkl")

class_threshold = model_metadata["classification_threshold"]
model_features = model_metadata["model_features"]

positive_class = model_metadata["positive_class"]
negative_class = model_metadata["negative_class"]

def create_dataframe(work_pressure, packers_assigned, bottleneck_flag):
    # Construct model input. Will be in the same format as what
    # was exported in the notebook FC_Operations_Risk_Predictions.ipynb
    input_data = pd.DataFrame([{
        "work_pressure": work_pressure,
        "packers_assigned": packers_assigned,
        "bottleneck_flag": bottleneck_flag
        }])

    # Enforce expected model features
    input_data = input_data[model_features]

    return input_data
    
def predict(current_backlog, planned_work, packers_assigned, bottleneck_flag):

    # If planned work was specified as 0, or not populated, it is
    # assumed there is no risk per business logic.
    
    if current_backlog < 0:
        raise ValueError("Current Backlog cannot be negative.")

    if planned_work < 0:
        raise ValueError("Planned Work cannot be negative.")

    if packers_assigned < 0:
        raise ValueError("Packers Assigned cannot be negative.")

    if bottleneck_flag not in [0, 1]:
        raise ValueError("Bottleneck must either be 0 or 1.")
    
    if planned_work == 0 and current_backlog == 0:
        probability = None
        prediction = model_metadata["negative_class"]

    elif packers_assigned == 0 and (backlog_units > 0 or planned_work > 0):
        probability = None
        prediction = model_metadata["positive_class"]
        
    else:
        # Engineer model feature. This calculation matches what appears
        # in the notebook FC_Operations_Risk_Predictions.ipynb
        work_pressure = current_backlog / planned_work
            
        input_data = create_dataframe(work_pressure, packers_assigned, bottleneck_flag)
            
        # Generate probability from the first (and only) row
        # for the positive_class (i.e. next_hour_backlog_risk = 1)
        probability = model_pipeline.predict_proba(input_data)[0, 1]
        
        # Apply deployment threshold
        if probability >= model_metadata["classification_threshold"]:
            prediction = model_metadata["positive_class"]
        else:
            prediction = model_metadata["negative_class"]
        
    return prediction, probability

