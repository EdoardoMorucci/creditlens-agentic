from __future__ import annotations

import os

# Ensure offline-friendly tests
os.environ.setdefault("FORCE_OFFLINE", "true")

from creditlens_agentic.service import resume_case, start_case
from creditlens_agentic.tools.cashflow import analyze_cashflow
from creditlens_agentic.tools.scoring import compute_potential_score
from creditlens_agentic.tools import load_persona, list_personas


def test_list_and_load_persona():
    personas = list_personas()
    assert "persona1" in personas
    data = load_persona("persona1")
    assert data["banking"].get("transactions")
    assert data["applicant"]["full_name"]


def test_cashflow_and_score_deterministic():
    data = load_persona("persona1")
    cashflow = analyze_cashflow(data["banking"])
    assert cashflow["cashflow_score"]["total_score"] > 0
    scores = compute_potential_score(
        cashflow,
        {"overall_score": 75},
        {"reputation_score": 8.5, "client_retention_rate": 0.4, "loan_applicability_score": 0.85},
    )
    assert 0 <= scores["general_score"] <= 1000
    assert scores["scoring_mode"] == "deterministic"


def test_agentic_case_hitl_flow():
    case = start_case("persona1", max_chat_rounds=2)
    assert case["thread_id"]
    assert case["interrupt"]["type"] == "applicant_question"
    assert case["cashflow"]["cashflow_score"]["total_score"] > 0

    case = resume_case(case["thread_id"], {"text": "Loan for equipment, about €8k. No other debts."})
    assert case["interrupt"]["type"] == "applicant_question"

    case = resume_case(case["thread_id"], {"text": "Main income is Stripe + Upwork retainers."})
    assert case["interrupt"]["type"] == "officer_review"
    assert case["scores"]["general_score"] > 0
    assert case["applicant_report"]

    case = resume_case(
        case["thread_id"],
        {"decision": "approve", "notes": "Strong freelancing profile"},
    )
    assert case["status"] == "completed"
    assert case["officer_decision"] == "approve"
    assert case["interrupt"] is None
