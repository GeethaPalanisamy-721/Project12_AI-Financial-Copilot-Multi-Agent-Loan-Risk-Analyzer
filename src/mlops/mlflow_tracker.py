# src/mlops/mlflow_tracker.py
"""
mlflow_tracker.py

PURPOSE:
--------
Tracks all ML experiments using MLflow.
Records:
- Model parameters
- Training metrics (ROC-AUC, F1)
- SHAP feature importance
- Trained model artifact

WHY MLFLOW?
-----------
Without MLflow → you forget which parameters gave best results
With MLflow    → every experiment is logged, compared, reproduced
"""

import mlflow
import mlflow.xgboost
import pandas as pd
import joblib
import json
import os


# ==============================
# 🔹 Setup MLflow
# ==============================
def setup_mlflow():
    """
    Sets local folder as MLflow tracking server.
    All runs saved to mlruns/ folder automatically.
    """
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("ai_financial_copilot")
    print("✅ MLflow experiment set: ai_financial_copilot")


# ==============================
# 🔹 Log Training Run
# ==============================
def log_training_run(
    params: dict,
    metrics: dict,
    model_path: str = "models/xgboost_model.pkl"
):
    """
    Logs one complete training run to MLflow.

    Parameters:
    -----------
    params     : dict — XGBoost hyperparameters
    metrics    : dict — ROC-AUC, F1, precision, recall
    model_path : str  — path to saved model

    Example:
    --------
    params  = {"n_estimators": 300, "max_depth": 5}
    metrics = {"roc_auc": 0.756, "f1_score": 0.27}
    """

    setup_mlflow()

    with mlflow.start_run(run_name="xgboost_loan_default"):

        # ==============================
        # 1. Log Parameters
        # What settings were used to train
        # ==============================
        print("\n📌 Logging parameters...")
        mlflow.log_params(params)

        # ==============================
        # 2. Log Metrics
        # How well the model performed
        # ==============================
        print("📊 Logging metrics...")
        mlflow.log_metrics(metrics)

        # ==============================
        # 3. Log SHAP Feature Importance
        # Which features mattered most
        # ==============================
        shap_path = "models/top_shap_features.json"
        if os.path.exists(shap_path):
            mlflow.log_artifact(shap_path)
            print("🔥 Logged SHAP features")

        # ==============================
        # 4. Log Classification Report
        # ==============================
        report_path = "reports/classification_report.txt"
        if os.path.exists(report_path):
            mlflow.log_artifact(report_path)
            print("📄 Logged classification report")

        # ==============================
        # 5. Log SHAP Plot
        # ==============================
        shap_plot = "reports/shap_bar.png"
        if os.path.exists(shap_plot):
            mlflow.log_artifact(shap_plot)
            print("📈 Logged SHAP plot")

        # ==============================
        # 6. Log Model
        # ==============================
        model = joblib.load(model_path)
        mlflow.xgboost.log_model(
            model,
            artifact_path="xgboost_model",
            registered_model_name="LoanDefaultPredictor"
        )
        print("✅ Model logged to MLflow")

        # Get run ID for reference
        run_id = mlflow.active_run().info.run_id
        print(f"\n🎯 MLflow Run ID: {run_id}")
        print("📂 View UI: mlflow ui")

    return run_id


# ==============================
# 🔹 Log Prediction (Inference)
# ==============================
def log_prediction(
    customer_data: dict,
    result: dict
):
    """
    Logs each prediction made via Streamlit.
    Useful for monitoring model in production.

    Parameters:
    -----------
    customer_data : dict — input features
    result        : dict — pipeline output
    """

    setup_mlflow()

    with mlflow.start_run(run_name="prediction_log"):

        # Log input features
        mlflow.log_params({
            "amt_credit"   : customer_data.get("AMT_CREDIT"),
            "amt_annuity"  : customer_data.get("AMT_ANNUITY"),
            "ext_source_2" : customer_data.get("EXT_SOURCE_2"),
            "ext_source_3" : customer_data.get("EXT_SOURCE_3"),
        })

        # Log prediction outputs
        mlflow.log_metrics({
            "default_probability": result["probability"],
            "dti_ratio"          : result["dti_ratio"]
        })

        mlflow.log_param("final_decision", 
                         result["final_decision"])
        mlflow.log_param("risk_level",
                         result["risk_level"])

        print(f"✅ Prediction logged to MLflow")


# ==============================
# ▶️ MAIN — Log Training Results
# Run: python -m src.mlops.mlflow_tracker
# ==============================
if __name__ == "__main__":

    # ✅ Your actual training parameters
    params = {
        "n_estimators"    : 300,
        "max_depth"       : 5,
        "learning_rate"   : 0.05,
        "subsample"       : 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 11.4,
        "eval_metric"     : "auc"
    }

    # ✅ Your actual results from train_model.py
    metrics = {
        "roc_auc"  : 0.7565,
        "threshold": 0.4
    }

    run_id = log_training_run(params, metrics)

    print("\n🎉 MLflow tracking complete!")
    print("👉 Run this to view dashboard:")
    print("   mlflow ui")
    print("   Then open: http://127.0.0.1:5000")