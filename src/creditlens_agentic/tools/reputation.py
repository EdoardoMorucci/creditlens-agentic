from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from creditlens_agentic.llm import get_llm


class ReputationAnalysis(BaseModel):
    reputation_score: float = Field(ge=0, le=10)
    reputation_summary: str
    client_retention_rate: float = Field(ge=0, le=1)
    sentiment_score: float = Field(ge=-1, le=1)
    overall_sentiment: str
    key_positive_themes: list[str]
    key_negative_themes: list[str]
    loan_applicability_score: float = Field(ge=0, le=1)
    loan_recommendation: str


def _fallback_reputation(raw: dict[str, Any]) -> dict[str, Any]:
    rating = float(raw.get("average_rating") or 4.0)
    success = float(raw.get("job_success_score") or 80)
    reviews = int(raw.get("total_reviews") or len(raw.get("feedback") or []))
    long_term = int(raw.get("long_term_clients") or 0)

    rep_score = min(10.0, (rating / 5) * 6 + (success / 100) * 3 + min(1.0, reviews / 50))
    retention = min(1.0, long_term / max(reviews, 1) * 3) if reviews else 0.3
    sentiment = max(-1.0, min(1.0, (rating - 3) / 2))
    loan_score = min(1.0, (rep_score / 10) * 0.7 + retention * 0.3)

    if loan_score >= 0.9:
        rec = "Highly Recommended for Freelance Income-Based Loan"
    elif loan_score >= 0.8:
        rec = "Recommended for Freelance Income-Based Loan"
    elif loan_score >= 0.7:
        rec = "Conditionally Recommended - Further Review Needed"
    else:
        rec = "Not Recommended"

    feedback = raw.get("feedback") or []
    themes = []
    for item in feedback[:5]:
        if isinstance(item, dict) and item.get("review"):
            themes.append(str(item["review"])[:80])

    return {
        "reputation_score": round(rep_score, 2),
        "reputation_summary": (
            f"{raw.get('name', 'Applicant')} — {raw.get('headline', 'freelancer')}. "
            f"Rating {rating}/5, job success {success}, {reviews} reviews."
        ),
        "client_retention_rate": round(retention, 3),
        "sentiment_score": round(sentiment, 3),
        "overall_sentiment": "Positive" if sentiment >= 0.3 else "Neutral" if sentiment >= -0.2 else "Negative",
        "key_positive_themes": themes[:3] or ["Reliable delivery"],
        "key_negative_themes": [],
        "loan_applicability_score": round(loan_score, 3),
        "loan_recommendation": rec,
        "mode": "offline_fallback",
    }


def analyze_reputation(raw: dict[str, Any]) -> dict[str, Any]:
    """LLM reputation analysis from marketplace profile, with fallback."""
    if not raw:
        return {
            "reputation_score": 0,
            "reputation_summary": "No reputation data",
            "client_retention_rate": 0,
            "sentiment_score": 0,
            "overall_sentiment": "Neutral",
            "key_positive_themes": [],
            "key_negative_themes": [],
            "loan_applicability_score": 0,
            "loan_recommendation": "Not Recommended",
            "mode": "missing_data",
        }

    llm = get_llm(temperature=0.2)
    if llm is None:
        return _fallback_reputation(raw)

    import json

    structured = llm.with_structured_output(ReputationAnalysis)
    prompt = f"""You are a credit risk analyst specializing in freelancer reputation.
Analyze this marketplace profile and return structured scores.

PROFILE JSON:
{json.dumps(raw, indent=2)[:10000]}
"""
    try:
        result: ReputationAnalysis = structured.invoke(prompt)
        data = result.model_dump()
        data["mode"] = "llm"
        return data
    except Exception as exc:  # noqa: BLE001
        fallback = _fallback_reputation(raw)
        fallback["llm_error"] = str(exc)
        return fallback
