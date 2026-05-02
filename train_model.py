"""
train_model.py
==============
Baby Food Quality Prediction — Ensemble Model Training
Trains a Voting Classifier (RF + XGBoost + Gradient Boosting)
and saves the model + scaler to disk.

Run:
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from xgboost import XGBClassifier

# ── 1. Load Data ────────────────────────────────────────────────────────
print("📦 Loading dataset...")
df = pd.read_csv("food_Ingredients.csv")
print(f"   Shape: {df.shape}")
print(f"   Quality distribution:\n{df['quality'].value_counts()}\n")

# ── 2. Preprocessing ────────────────────────────────────────────────────
# Drop rows with null target
df = df.dropna(subset=["quality"])

# Drop non-feature columns
DROP_COLS = ["Unnamed: 0", "quality"]
feature_cols = [c for c in df.columns if c not in DROP_COLS]

X = df[feature_cols].copy()
y = df["quality"].astype(int).copy()

# Fill missing numeric values with median
X = X.fillna(X.median(numeric_only=True))

# Save feature column names for inference
joblib.dump(feature_cols, "feature_cols.pkl")
print(f"✅ Feature columns saved ({len(feature_cols)} features)")

# ── 3. Train / Test Split ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"   Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows\n")

# ── 4. Scaling ──────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved → scaler.pkl")

# ── 5. Remap labels for XGBoost (needs 0-based continuous labels) ───────
unique_labels = sorted(y.unique())
label_to_idx  = {lbl: idx for idx, lbl in enumerate(unique_labels)}
idx_to_label  = {idx: lbl for lbl, idx in label_to_idx.items()}
joblib.dump(idx_to_label, "idx_to_label.pkl")

y_train_enc = y_train.map(label_to_idx)
y_test_enc  = y_test.map(label_to_idx)

# ── 6. Define Base Models ───────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
)

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

gbm = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
)

# ── 7. Voting Classifier ────────────────────────────────────────────────
print("🧠 Training Ensemble (RF + XGBoost + GBM) — this may take a minute...")
ensemble = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb), ("gbm", gbm)],
    voting="soft",
)
ensemble.fit(X_train_sc, y_train_enc)
print("✅ Model trained successfully!\n")

# ── 8. Evaluation ───────────────────────────────────────────────────────
y_pred_enc = ensemble.predict(X_test_sc)

# Map back to original labels for reporting
y_test_orig = y_test_enc.map(idx_to_label)
y_pred_orig = pd.Series(y_pred_enc).map(idx_to_label)

print("=" * 55)
print("📊 CLASSIFICATION REPORT")
print("=" * 55)
print(classification_report(y_test_orig, y_pred_orig))

f1 = f1_score(y_test_orig, y_pred_orig, average="weighted")
print(f"🏆 Weighted F1 Score: {f1:.4f}")

print("\n📉 CONFUSION MATRIX")
cm = confusion_matrix(y_test_orig, y_pred_orig, labels=unique_labels)
cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in unique_labels],
                         columns=[f"Pred {l}" for l in unique_labels])
print(cm_df)
print("=" * 55)

# ── 9. Save Model ───────────────────────────────────────────────────────
joblib.dump(ensemble, "food_quality_model.pkl")
print("\n✅ Model saved → food_quality_model.pkl")
print("✅ Label mapping saved → idx_to_label.pkl")
print("\n🚀 All done! You can now run: streamlit run app.py")
