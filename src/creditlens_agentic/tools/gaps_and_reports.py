from __future__ import annotations

import json
from typing import Any

from creditlens_agentic.llm import get_llm


def detect_gaps(
    cashflow: dict[str, Any],
    document_analysis: dict[str, Any],
    reputation: dict[str, Any],
    banking: dict[str, Any],
) -> list[str]:
    """Heuristic gap detection — what the chat agent should ask about."""
    gaps: list[str] = []

    txs = (banking or {}).get("transactions") or []
    if len(txs) < 5:
        gaps.append("Limited bank transaction history — ask about other income accounts")

    concerns = ((cashflow or {}).get("insights") or {}).get("concerns") or []
    for c in concerns:
        gaps.append(f"Cashflow concern: {c}")

    if float((document_analysis or {}).get("overall_score") or 0) < 60:
        gaps.append("Weak document evidence — ask for recent paid invoices or contracts")

    if float((reputation or {}).get("reputation_score") or 0) < 6:
        gaps.append("Soft reputation signals — ask about long-term clients and pipeline")

    if not gaps:
        gaps.append("Confirm loan purpose and requested amount")
        gaps.append("Confirm existing debt obligations")

    return gaps[:6]


def generate_reports(
    applicant: dict[str, Any],
    cashflow: dict[str, Any],
    document_analysis: dict[str, Any],
    reputation: dict[str, Any],
    scores: dict[str, Any],
    chat_summary: str = "",
) -> tuple[str, str]:
    """Generate applicant + analyst narrative reports."""
    payload = {
        "applicant": applicant,
        "cashflow": cashflow,
        "document_analysis": document_analysis,
        "reputation": reputation,
        "scores": scores,
        "chat_summary": chat_summary,
    }

    llm = get_llm(temperature=0.3)
    if llm is None:
        return _fallback_reports(payload)

    try:
        applicant_report = llm.invoke(
            [
                (
                    "system",
                    "You write clear, encouraging credit assessment summaries for freelancers "
                    "(150-250 words). Never guarantee approval. Use markdown.",
                ),
                ("human", f"Write the applicant-facing report from:\n{json.dumps(payload, indent=2)[:10000]}"),
            ]
        ).content
        analyst_report = llm.invoke(
            [
                (
                    "system",
                    "You are a senior credit analyst. Write a detailed lending memo (600-900 words) "
                    "with executive summary, financials, risks, recommendation. Use markdown.",
                ),
                ("human", f"Write the analyst report from:\n{json.dumps(payload, indent=2)[:10000]}"),
            ]
        ).content
        return str(applicant_report).strip(), str(analyst_report).strip()
    except Exception:  # noqa: BLE001
        return _fallback_reports(payload)


def _fallback_reports(payload: dict[str, Any]) -> tuple[str, str]:
    name = (payload.get("applicant") or {}).get("full_name", "Applicant")
    scores = payload.get("scores") or {}
    general = scores.get("general_score", "N/A")
    rec = scores.get("recommendation", "N/A")
    risk = scores.get("risk_level", "N/A")

    applicant = f"""# Your CreditLens Potential Score

Hello {name},

Your **Potential Score** is **{general}/1000** (risk: {risk}).

**Recommendation:** {rec}

This assessment combines cashflow health, verified income documents, and professional reputation.
Scores are calculated with a fixed formula — the AI explains results but does not invent the number.

Next steps: a loan officer will review the file. You may be asked for additional documentation.
"""

    cf = (payload.get("cashflow") or {}).get("cashflow_score") or {}
    analyst = f"""# Analyst Memo — {name}

## Executive Summary
- General score: **{general}/1000**
- Risk: **{risk}**
- Recommendation: **{rec}**
- Max loan: {scores.get("max_loan_amount")}
- Suggested rate: {scores.get("suggested_interest_rate")}

## Score Breakdown
- Financial health: {scores.get("financial_health_score")}
- Future income: {scores.get("future_income_score")}
- Reputation: {scores.get("reputation_score")}

## Cashflow
- Total: {cf.get("total_score")}/100 ({cf.get("rating")})

## Notes
Deterministic scoring path (offline or LLM unavailable). Review chat transcript for qualitative context.
"""
    return applicant.strip(), analyst.strip()
