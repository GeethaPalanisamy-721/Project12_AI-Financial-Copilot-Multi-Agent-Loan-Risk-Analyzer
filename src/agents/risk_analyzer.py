# src/agents/risk_analyzer.py
"""
risk_analyzer.py

PURPOSE:
--------
Calculates DTI (Debt-to-Income) ratio from customer data.
Returns risk band: HIGH / MEDIUM / LOW

DTI = AMT_CREDIT / AMT_ANNUITY
Higher DTI = harder to repay = higher risk
"""

def risk_analysis(sample: dict) -> dict:
    """
    Parameters:
    -----------
    sample : dict — customer data

    Returns:
    --------
    dict:
        dti_ratio  : float
        risk_level : str (HIGH / MEDIUM / LOW)
    """

    credit  = sample.get("AMT_CREDIT", 0)
    annuity = sample.get("AMT_ANNUITY", 1)

    # Avoid division by zero
    dti_ratio = credit / (annuity + 1)

    # ✅ Risk bands aligned with probability thresholds
    if dti_ratio > 50:
        risk_level = "HIGH"
    elif dti_ratio > 17:       # ← lowered from 20 to 17
        risk_level = "MEDIUM"  #   catches more borderline cases
    else:
        risk_level = "LOW"

    return {
        "dti_ratio" : round(dti_ratio, 2),
        "risk_level": risk_level
    }