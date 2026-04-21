# src/orchestrator.py
"""
orchestrator.py

PURPOSE:
--------
Connects all 4 agents in sequence.
Streamlit calls ONLY this file.

FLOW:
-----
Customer Input
     ↓
Classifier Agent  → probability + risk label
     ↓
Risk Analyzer     → DTI ratio + risk band
     ↓
Decision Agent    → APPROVE / REJECT / REVIEW
     ↓
Explainer Agent   → plain English explanation
     ↓
Result Dict       → returned to Streamlit
"""

import sys
import os
sys.path.append(os.path.abspath("."))

import pandas as pd

from src.agents.classifier_agent import classify_customer
from src.agents.risk_analyzer    import risk_analysis
from src.agents.decision_agent   import final_decision
from src.agents.explainer_agent  import generate_explanation


def run_pipeline(customer_data: dict) -> dict:
    """
    Master pipeline function.

    Parameters:
    -----------
    customer_data : dict — from Streamlit form

    Returns:
    --------
    dict with all results for Streamlit to display
    """

    print("\n🚀 Pipeline Started...")
    print(f"📥 Features received: {len(customer_data)}")

    # ==============================
    # STEP 1 — Classifier Agent
    # XGBoost predicts default probability
    # ==============================
    print("\n⚙️  Step 1: Classifier Agent")
    prob, model_decision = classify_customer(
        pd.Series(customer_data))
    print(f"   Probability    : {prob:.4f}")
    print(f"   Model Decision : {model_decision}")

    # ==============================
    # STEP 2 — Risk Analyzer
    # DTI ratio based risk banding
    # ==============================
    print("\n⚙️  Step 2: Risk Analyzer")
    risk = risk_analysis(customer_data)
    print(f"   DTI Ratio  : {risk['dti_ratio']}")
    print(f"   Risk Level : {risk['risk_level']}")

    # ==============================
    # STEP 3 — Decision Agent
    # Combines Steps 1+2 → business decision
    # Pass prob + model_decision to avoid recomputing
    # ==============================
    print("\n⚙️  Step 3: Decision Agent")
    decision_output = final_decision(
        customer_data,
        prob=prob,
        model_decision=model_decision
    )
    print(f"   Risk Level (final) : {decision_output['risk_level']}")
    print(f"   Final Decision     : {decision_output['final_decision']}")

    # ==============================
    # STEP 4 — Explainer Agent
    # Build query from ACTUAL decision output
    # ✅ Explanation matches real decision — not hardcoded
    # ==============================
    print("\n⚙️  Step 4: Explainer Agent")
    query = (
        f"Customer classified as {decision_output['model_decision']} "
        f"with probability {prob:.2f}. "
        f"Risk level is {decision_output['risk_level']}. "
        f"Final decision is {decision_output['final_decision']}."
    )
    explanation = generate_explanation(query)
    print("   Explanation Generated ✅")

    # ==============================
    # STEP 5 — Combine all outputs
    # Single dict returned to Streamlit
    # ==============================
    result = {
        "probability"   : round(prob, 4),
        "model_decision": decision_output["model_decision"],
        "risk_level"    : decision_output["risk_level"],
        "dti_ratio"     : decision_output["dti_ratio"],
        "final_decision": decision_output["final_decision"],
        "explanation"   : explanation
    }

    print("\n✅ Pipeline Complete!")
    print("=" * 40)
    return result


# ==============================
# 🧪 Quick Test
# Run: python -m src.orchestrator
# ==============================
if __name__ == "__main__":

    # Test with borderline customer
    sample = {
        "AMT_CREDIT"         : 500000,
        "AMT_ANNUITY"        : 25000,
        "EXT_SOURCE_2"       : 0.3,
        "EXT_SOURCE_3"       : 0.25,
        "DAYS_EMPLOYED"      : -500,
        "DAYS_BIRTH"         : -12000,
        "AMT_GOODS_PRICE"    : 450000,
        "CODE_GENDER"        : 1,
        "FLAG_OWN_CAR"       : 0,
        "NAME_EDUCATION_TYPE": 1
    }

    result = run_pipeline(sample)

    print("\n📊 FINAL OUTPUT:")
    print("=" * 40)
    for k, v in result.items():
        if k != "explanation":
            print(f"{k:20s}: {v}")
    print(f"\n💬 EXPLANATION:\n{result['explanation']}")