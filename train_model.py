# train_model.py

import warnings
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import FunctionTransformer
import joblib

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

# ----------------------------
# Step 3: Preprocessing setup
# ----------------------------
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
    ('flatten', FunctionTransformer(lambda x: x.ravel(), validate=False)),  # ✅ fix here
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
print(f"CV accuracy (mean±std): {scores.mean():.4f} ± {scores.std():.4f}")

pipe.fit(train_df[features], train_df[target])
preds = pipe.predict(test_df[features])
print("Test accuracy:", accuracy_score(test_df[target], preds))
print("\nClassification report:\n", classification_report(test_df[target], preds))

# ----------------------------
# Step 6: Save model artifacts
# ----------------------------
Path("model").mkdir(exist_ok=True)
joblib.dump(pipe, "model/healthcare_model.joblib")
train_df.to_csv("model/train_data.csv", index=False)
test_df.to_csv("model/test_data.csv", index=False)
print("✅ Model and data saved successfully.")
