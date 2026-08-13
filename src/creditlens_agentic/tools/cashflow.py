from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _to_dataframe(banking: dict[str, Any]) -> pd.DataFrame:
    txs = banking.get("transactions") or []
    if not txs:
        return pd.DataFrame()

    rows = []
    for tx in txs:
        amount = float(tx.get("amount", 0))
        rows.append(
            {
                "transaction_id": tx.get("transaction_id") or tx.get("id"),
                "amount": amount,
                "date": pd.to_datetime(tx.get("date"), errors="coerce"),
                "merchant": tx.get("merchant") or tx.get("description") or "",
                "category": tx.get("category", ""),
                "classification": tx.get("classification", ""),
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["date", "amount"])
    if df.empty:
        return df
    df["is_credit"] = df["amount"] > 0
    df["is_debit"] = df["amount"] < 0
    df["abs_amount"] = df["amount"].abs()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df.sort_values("date").reset_index(drop=True)


def analyze_cashflow(banking: dict[str, Any]) -> dict[str, Any]:
    """Deterministic cashflow scoring from Open Banking-style transactions."""
    df = _to_dataframe(banking)
    if df.empty:
        return {
            "cashflow_score": {
                "balance_trend": 0,
                "income_stability": 0,
                "expense_control": 0,
                "liquidity": 0,
                "total_score": 0,
                "rating": "INSUFFICIENT_DATA",
            },
            "financial_metrics": {},
            "risk_assessment": {},
            "insights": {
                "strengths": [],
                "concerns": ["No transaction data available"],
            },
        }

    df = df.copy()
    df["running_balance"] = df["amount"].cumsum()

    # Balance trend (0-30)
    if len(df) > 1:
        trend = float(np.polyfit(range(len(df)), df["running_balance"], 1)[0])
        balance_score = float(min(30, max(0, (trend / 100) + 15)))
    else:
        balance_score = 15.0

    # Income stability (0-25)
    monthly_income = df[df["is_credit"]].groupby(["year", "month"])["amount"].sum()
    if len(monthly_income) > 1 and monthly_income.mean() > 0:
        cv = float(monthly_income.std() / monthly_income.mean())
        income_stability = float(max(0, 25 - (cv * 25)))
    else:
        income_stability = 15.0

    # Expense control (0-25)
    months = max(int(df["month"].nunique()), 1)
    avg_income = float(df[df["is_credit"]]["amount"].sum() / months) or 1.0
    monthly_expenses = df[df["is_debit"]].groupby(["year", "month"])["amount"].sum().abs()
    avg_expenses = float(monthly_expenses.mean()) if len(monthly_expenses) else 0.0
    expense_ratio = avg_expenses / avg_income if avg_income else 1.0
    expense_control = float(max(0, 25 - (expense_ratio * 25)))

    # Liquidity (0-20)
    current_balance = float(df["running_balance"].iloc[-1])
    avg_monthly_exp = float(df[df["is_debit"]]["amount"].abs().sum() / months) or 1.0
    liquidity_months = current_balance / avg_monthly_exp
    liquidity = float(min(20, max(0, liquidity_months * 5)))

    total = balance_score + income_stability + expense_control + liquidity
    if total >= 80:
        rating = "EXCELLENT"
    elif total >= 65:
        rating = "GOOD"
    elif total >= 50:
        rating = "FAIR"
    else:
        rating = "POOR"

    total_income = float(df[df["is_credit"]]["amount"].sum())
    total_expenses = float(df[df["is_debit"]]["amount"].abs().sum())
    net = total_income - total_expenses
    savings_rate = (net / total_income * 100) if total_income else 0.0

    income_vol = (
        float(df[df["is_credit"]]["amount"].std() / df[df["is_credit"]]["amount"].mean())
        if df[df["is_credit"]]["amount"].mean()
        else 0.0
    )
    risk_score = float(min(100, max(0, income_vol * 40 + (100 - total) * 0.4)))

    strengths: list[str] = []
    concerns: list[str] = []
    if income_stability >= 18:
        strengths.append("Stable recurring income pattern")
    else:
        concerns.append("Income volatility across months")
    if savings_rate >= 10:
        strengths.append(f"Positive savings rate ({savings_rate:.1f}%)")
    else:
        concerns.append("Low or negative savings rate")
    if liquidity >= 10:
        strengths.append("Healthy liquidity buffer")
    else:
        concerns.append("Thin liquidity cushion")

    return {
        "cashflow_score": {
            "balance_trend": round(balance_score, 2),
            "income_stability": round(income_stability, 2),
            "expense_control": round(expense_control, 2),
            "liquidity": round(liquidity, 2),
            "total_score": round(total, 2),
            "rating": rating,
        },
        "financial_metrics": {
            "total_income": round(total_income, 2),
            "total_expenses": round(total_expenses, 2),
            "net_cashflow": round(net, 2),
            "savings_rate": round(savings_rate, 2),
            "avg_monthly_income": round(avg_income, 2),
            "avg_monthly_expenses": round(avg_expenses, 2),
            "current_balance": round(current_balance, 2),
            "transaction_count": int(len(df)),
        },
        "risk_assessment": {
            "income_volatility": round(income_vol, 3),
            "overall_risk_score": round(risk_score, 2),
        },
        "insights": {"strengths": strengths, "concerns": concerns},
    }
