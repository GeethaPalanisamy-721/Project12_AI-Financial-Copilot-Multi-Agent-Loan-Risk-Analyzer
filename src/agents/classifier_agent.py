# src/agents/classifier_agent.py
"""
classifier_agent.py

PURPOSE:
--------
Loads trained XGBoost model.
Predicts default probability for a given customer.
Returns probability + risk label.
"""

import joblib
import pandas as pd

# ✅ Load model ONCE at module level — not inside function
# This avoids reloading model on every prediction call
model = joblib.load("models/xgboost_model.pkl")

# ✅ Load exact training columns — CRITICAL
# Model expects EXACT same columns in EXACT same order
feature_columns = pd.read_csv(
    "data/processed/X_train.csv", nrows=1).columns


def classify_customer(sample: pd.Series) -> tuple:
    """
    Parameters:
    -----------
    sample : pd.Series — one customer row

    Returns:
    --------
    tuple: (probability: float, decision: str)
        decision → "HIGH RISK" / "MEDIUM RISK" / "LOW RISK"
    """

    # Convert Series to DataFrame (model expects 2D)
    sample_df = pd.DataFrame([sample])

    # ✅ Align columns — fill missing features with 0
    # This prevents errors when Streamlit sends partial input
    sample_df = sample_df.reindex(
        columns=feature_columns, fill_value=0)

    # Get default probability (class 1 = default)
    prob = model.predict_proba(sample_df)[0][1]

    # ✅ 3-band classification (more meaningful than 2)
    if prob >= 0.6:
        decision = "HIGH RISK"
    elif prob >= 0.35:
        decision = "MEDIUM RISK"
    else:
        decision = "LOW RISK"

    return prob, decision