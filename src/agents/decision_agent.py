# src/agents/decision_agent.py
"""
decision_agent.py

PURPOSE:
--------
Combines ML prediction + DTI risk analysis.
Applies business rules to produce final decision.

DECISION RULES:
---------------
REJECT  → prob >= 0.6  OR  risk = HIGH
APPROVE → prob < 0.35  AND risk = LOW
REVIEW  → everything else (borderline cases)
"""

from src.agents.risk_analyzer import risk_analysis
import pandas as pd


def final_decision(sample: dict,
                   prob: float = None,
                   model_decision: str = None) -> dict:
    """
    Parameters:
    -----------
    sample         : dict  — customer data
    prob           : float — from classifier (optional)
    model_decision : str   — from classifier (optional)

    Returns:
    --------
    dict:
        probability, model_decision, risk_level,
        dti_ratio, final_decision
    """

    # ==============================
    # STEP 1: Get probability if not passed
    # Orchestrator passes prob directly to avoid double computation
    # ==============================
    if prob is None or model_decision is None:
        from src.agents.classifier_agent import classify_customer
        prob, model_decision = classify_customer(
            pd.Series(sample))

    # ==============================
    # STEP 2: Get DTI risk level
    # ==============================
    risk = risk_analysis(sample)
    risk_level = risk["risk_level"]

    # ==============================
    # STEP 3: Override risk level if probability
    # contradicts DTI-based risk
    # Example: prob=0.56 but DTI says LOW → upgrade to MEDIUM
    # ==============================
    if prob >= 0.6 and risk_level != "HIGH":
        risk_level = "HIGH"       # ← upgrade LOW/MEDIUM to HIGH
    elif prob >= 0.35 and risk_level == "LOW":
        risk_level = "MEDIUM"     # ← upgrade LOW to MEDIUM

    # ==============================
    # STEP 4: Business Decision Rules
    # ==============================
    if prob >= 0.6 or risk_level == "HIGH":
        decision = "REJECT"

    elif prob < 0.35 and risk_level == "LOW":
        decision = "APPROVE"

    else:
        decision = "REVIEW"       # ← all borderline cases

    return {
        "probability"   : round(prob, 4),
        "model_decision": model_decision,
        "risk_level"    : risk_level,
        "dti_ratio"     : risk["dti_ratio"],
        "final_decision": decision
    }