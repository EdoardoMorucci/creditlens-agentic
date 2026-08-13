from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from creditlens_agentic.llm import get_llm


class DocumentAnalysis(BaseModel):
    overall_score: float = Field(ge=0, le=100)
    risk_rating: str
    financial_stability_score: float = Field(ge=0, le=100)
    payment_history_score: float = Field(ge=0, le=100)
    executive_summary: str
    risk_factors: list[str]
    positive_indicators: list[str]
    recommendations: list[str]
    confidence_level: str


def _fallback_document_analysis(documents_text: str, invoices: dict[str, Any]) -> dict[str, Any]:
    inv_list = invoices.get("invoices") or invoices.get("items") or []
    if isinstance(invoices, list):
        inv_list = invoices

    count = len(inv_list) if isinstance(inv_list, list) else 0
    paid = 0
    total_amount = 0.0
    if isinstance(inv_list, list):
        for inv in inv_list:
            if not isinstance(inv, dict):
                continue
            status = str(inv.get("status", "")).lower()
            if status in {"paid", "settled", "completed"}:
                paid += 1
            amount = inv.get("amount") or inv.get("total") or inv.get("value") or 0
            try:
                total_amount += float(amount)
            except (TypeError, ValueError):
                pass

    pay_ratio = (paid / count) if count else 0.5
    payment_score = 50 + pay_ratio * 45
    stability = min(100, 55 + (count * 5) + (10 if total_amount > 10000 else 0))
    overall = round((payment_score + stability) / 2, 1)
    risk = "LOW" if overall >= 80 else "MEDIUM" if overall >= 60 else "HIGH"

    return {
        "overall_score": overall,
        "risk_rating": risk,
        "financial_stability_score": round(stability, 1),
        "payment_history_score": round(payment_score, 1),
        "executive_summary": (
            f"Deterministic document pass over {count} invoices "
            f"(~€{total_amount:,.0f} total). Payment ratio={pay_ratio:.0%}."
        ),
        "risk_factors": [] if pay_ratio >= 0.8 else ["Incomplete payment history on invoices"],
        "positive_indicators": (
            ["Strong invoice volume"] if count >= 3 else ["Limited invoice sample"]
        ),
        "recommendations": ["Verify outstanding receivables with applicant"],
        "confidence_level": "MEDIUM" if documents_text else "LOW",
        "mode": "offline_fallback",
    }


def analyze_documents(documents_text: str, invoices: dict[str, Any] | None = None) -> dict[str, Any]:
    """LLM structured analysis of invoices/docs, with deterministic fallback."""
    invoices = invoices or {}
    llm = get_llm(temperature=0.1)
    if llm is None or not documents_text.strip():
        return _fallback_document_analysis(documents_text, invoices)

    structured = llm.with_structured_output(DocumentAnalysis)
    prompt = f"""You are a credit analyst reviewing freelancer invoices and payment records.
Produce a structured credit-oriented document analysis.

DOCUMENTS:
{documents_text[:12000]}
"""
    try:
        result: DocumentAnalysis = structured.invoke(prompt)
        data = result.model_dump()
        data["mode"] = "llm"
        return data
    except Exception as exc:  # noqa: BLE001
        fallback = _fallback_document_analysis(documents_text, invoices)
        fallback["llm_error"] = str(exc)
        return fallback
