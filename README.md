---
title: Healthcare Disease Predictor
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.47.0"
app_file: app.py
pinned: false
---

## 🧠 Features
- Uses real healthcare dataset (`health_chatbot_structured_features.csv`)
- Trains RandomForestClassifier with TF-IDF + OneHot + Median Imputation pipeline
- Auto-saves `.joblib` model
- Interactive Gradio interface for disease prediction

## 🚀 CI/CD with GitHub Actions
Whenever you push changes, GitHub Actions automatically:
1. Runs `train_model.py`
2. Saves trained model (`.joblib`)
3. Pushes the updated model and app to your Hugging Face Space
