# app.py

import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os

# Load the trained model
model_path = "model/healthcare_model.joblib"
if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model not found! Please run train_model.py first.")

pipe = joblib.load(model_path)

def predict_top_k(symptoms_text, duration_days, severity):
    row = {'symptoms_text': symptoms_text, 'duration_days_reported': duration_days, 'severity_level': severity}
    X = pd.DataFrame([row])
    proba = pipe.predict_proba(X)[0]
    classes = pipe.classes_
    idx = np.argsort(proba)[::-1][:3]
    return [{'disease': classes[i], 'probability': float(proba[i])} for i in idx]

iface = gr.Interface(
    fn=predict_top_k,
    inputs=[
        gr.Textbox(label="Symptoms Text"),
        gr.Number(label="Duration (days)"),
        gr.Dropdown(label="Severity Level", choices=['mild', 'moderate', 'severe'], value='mild')
    ],
    outputs=gr.JSON(label="Top 3 Predicted Diseases"),
    title="🩺 Healthcare Disease Prediction",
    description="Enter symptoms and details to get top disease predictions."
)

if __name__ == "__main__":
    iface.launch()
