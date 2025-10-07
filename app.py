import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os
from huggingface_hub import hf_hub_download

# -----------------------------
# Step 1: Download model from Hugging Face Hub
# ----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")  # GitHub Actions or Space secret
REPO_ID = "udaysankarjalli/healthcare-disease-predictor-model"
MODEL_FILENAME = "healthcare_model.joblib"

try:
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
        token=HF_TOKEN
    )
    print(f"✅ Model downloaded successfully: {model_path}")
except Exception as e:
    raise FileNotFoundError(f"Failed to download model from HF Hub: {e}")

# Load the trained pipeline
pipe = joblib.load(model_path)

# ----------------------------
# Step 2: Prediction function
# ----------------------------
def predict_top_k(symptoms_text, duration_days, severity):
    row = {
        'symptoms_text': symptoms_text,
        'duration_days_reported': duration_days,
        'severity_level': severity
    }
    X = pd.DataFrame([row])
    proba = pipe.predict_proba(X)[0]
    classes = pipe.classes_
    idx = np.argsort(proba)[::-1][:3]
    return [{'disease': classes[i], 'probability': float(proba[i])} for i in idx]

# ----------------------------
# Step 3: Gradio Interface
# ----------------------------
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
