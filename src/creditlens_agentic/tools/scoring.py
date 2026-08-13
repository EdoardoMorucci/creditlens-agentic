from __future__ import annotations

from typing import Any


def compute_potential_score(
    cashflow: dict[str, Any],
    document_analysis: dict[str, Any],
    reputation: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic Potential Score — LLM must not invent the final score."""

    cf = (cashflow or {}).get("cashflow_score") or {}
    cashflow_total = float(cf.get("total_score") or 50)

    # Normalize cashflow (0-100) → financial health (0-1000)
    financial_health = min(1000.0, max(0.0, cashflow_total * 10))

    # Document contribution blended into financial health
    doc_score = float((document_analysis or {}).get("overall_score") or 50)
    financial_health = (financial_health * 0.7) + (doc_score * 10 * 0.3)

    # Reputation: 0-10 → 0-1000
    rep_0_10 = float((reputation or {}).get("reputation_score") or 5)
    reputation_score = min(1000.0, max(0.0, (rep_0_10 / 10) * 1000))

    # Future income from retention + cashflow stability + loan applicability
    retention = float((reputation or {}).get("client_retention_rate") or 0.3)
    loan_app = float((reputation or {}).get("loan_applicability_score") or 0.5)
    income_stability = float(cf.get("income_stability") or 12)  # out of 25
    future_income = min(
        1000.0,
        max(
            0.0,
            retention * 350
            + loan_app * 350
            + (income_stability / 25) * 300,
        ),
    )

    general = financial_health * 0.40 + future_income * 0.30 + reputation_score * 0.30

    if general >= 800:
        risk_level, recommendation = "Low", "Highly Recommended"
        max_loan, rate = "Up to €50,000", "Prime + 2-4%"
    elif general >= 650:
        risk_level, recommendation = "Medium-Low", "Recommended"
        max_loan, rate = "Up to €30,000", "Prime + 4-6%"
    elif general >= 500:
        risk_level, recommendation = "Medium", "Conditionally Recommended"
        max_loan, rate = "Up to €15,000", "Prime + 6-8%"
    elif general >= 350:
        risk_level, recommendation = "Medium-High", "Not Recommended - High Risk"
        max_loan, rate = "Up to €5,000", "Prime + 8-12%"
    else:
        risk_level, recommendation = "High", "Not Recommended"
        max_loan, rate = "N/A", "N/A"

    return {
        "general_score": round(general, 1),
        "financial_health_score": round(financial_health, 1),
        "future_income_score": round(future_income, 1),
        "reputation_score": round(reputation_score, 1),
        "weights": {"financial_health": 0.40, "future_income": 0.30, "reputation": 0.30},
        "risk_level": risk_level,
        "recommendation": recommendation,
        "max_loan_amount": max_loan,
        "suggested_interest_rate": rate,
        "scoring_mode": "deterministic",
    }
