import warnings
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from huggingface_hub import HfApi

# ----------------------------
# Step 1: Load dataset
# ----------------------------
df = pd.read_csv("health_chatbot_structured_features.csv")
print("Data shape:", df.shape)

# ----------------------------
# Step 2: Define features & target
# ----------------------------
features = ['symptoms_text', 'duration_days_reported', 'severity_level']
target = 'disease_label'

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])
print("Train:", train_df.shape, " Test:", test_df.shape)

# ----------------------------
# Step 3: Preprocessing setup
# ----------------------------

def flatten_text(x):
    return x.ravel()

numeric_features = ['duration_days_reported']
numeric_transformer = SimpleImputer(strategy='median')

categorical_features = ['severity_level']
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

text_feature = 'symptoms_text'
text_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='')),
    ('flatten', FunctionTransformer(flatten_text, validate=False)),
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_df=0.95))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
    ('text', text_transformer, [text_feature])
])

# ----------------------------
# Step 4: Model Pipeline
# ----------------------------
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
])

# ----------------------------
# Step 5: Train & Evaluate
# ----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, train_df[features], train_df[target], cv=cv, scoring='accuracy', n_jobs=-1)
print(f"\nCV accuracy (mean ± std): {scores.mean():.4f} ± {scores.std():.4f}")

pipe.fit(train_df[features], train_df[target])
preds = pipe.predict(test_df[features])
print("\nTest accuracy:", accuracy_score(test_df[target], preds))
print("\nClassification report:\n", classification_report(test_df[target], preds))

# ----------------------------
# Step 6: Save model artifacts
# ----------------------------
Path("model").mkdir(exist_ok=True)

model_path = "model/healthcare_model.joblib"
train_path = "model/train_data.csv"
test_path = "model/test_data.csv"

joblib.dump(pipe, model_path)
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("\n✅ Model and data saved successfully:")
print(f"   Model  → {model_path}")
print(f"   Train  → {train_path}")
print(f"   Test   → {test_path}")

# ----------------------------
# Step 7: Upload model to Hugging Face Hub
# ----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")   # GitHub Action secret
REPO_ID = "udaysankarjalli/healthcare-disease-predictor-model"  # separate repo for large model files

if HF_TOKEN:
    api = HfApi()
    # Upload model
    api.upload_file(path_or_fileobj=model_path,
                    path_in_repo="healthcare_model.joblib",
                    repo_id=REPO_ID,
                    token=HF_TOKEN)
    # Upload train/test CSVs (optional)
    api.upload_file(path_or_fileobj=train_path,
                    path_in_repo="train_data.csv",
                    repo_id=REPO_ID,
                    token=HF_TOKEN)
    api.upload_file(path_or_fileobj=test_path,
                    path_in_repo="test_data.csv",
                    repo_id=REPO_ID,
                    token=HF_TOKEN)
    print("✅ Model and data uploaded successfully to Hugging Face Hub!")
else:
    print("⚠️ HF_TOKEN not found. Skipping upload.")
