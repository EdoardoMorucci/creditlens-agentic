from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from creditlens_agentic.llm import get_llm
from creditlens_agentic.state import CaseState
from creditlens_agentic.tools import load_persona
from creditlens_agentic.tools.cashflow import analyze_cashflow
from creditlens_agentic.tools.documents import analyze_documents
from creditlens_agentic.tools.gaps_and_reports import detect_gaps, generate_reports
from creditlens_agentic.tools.reputation import analyze_reputation
from creditlens_agentic.tools.scoring import compute_potential_score


def ingest_node(state: CaseState) -> dict[str, Any]:
    persona_id = state.get("persona_id") or "persona1"
    data = load_persona(persona_id)
    return {
        "case_id": state.get("case_id") or str(uuid.uuid4()),
        "persona_id": persona_id,
        "applicant": data["applicant"],
        "banking": data["banking"],
        "invoices": data["invoices"],
        "payments": data["payments"],
        "reputation_raw": data["reputation_raw"],
        "documents_text": data["documents_text"],
        "chat_rounds": 0,
        "max_chat_rounds": state.get("max_chat_rounds") or 3,
        "awaiting_human": False,
        "status": "running",
        "messages": [],
    }


def analyze_all_node(state: CaseState) -> dict[str, Any]:
    """Run deterministic + LLM analysis tools (same signals as CreditLens)."""
    cashflow = analyze_cashflow(state.get("banking") or {})
    document_analysis = analyze_documents(
        state.get("documents_text") or "",
        state.get("invoices") or {},
    )
    reputation = analyze_reputation(state.get("reputation_raw") or {})
    return {
        "cashflow": cashflow,
        "document_analysis": document_analysis,
        "reputation": reputation,
    }


def gap_node(state: CaseState) -> dict[str, Any]:
    gaps = detect_gaps(
        state.get("cashflow") or {},
        state.get("document_analysis") or {},
        state.get("reputation") or {},
        state.get("banking") or {},
    )
    return {"gaps": gaps}


def _chat_system_prompt(state: CaseState) -> str:
    ctx = {
        "applicant": state.get("applicant"),
        "cashflow_score": (state.get("cashflow") or {}).get("cashflow_score"),
        "document_analysis": {
            k: (state.get("document_analysis") or {}).get(k)
            for k in ("overall_score", "risk_rating", "executive_summary")
        },
        "reputation": {
            k: (state.get("reputation") or {}).get(k)
            for k in ("reputation_score", "loan_recommendation", "reputation_summary")
        },
        "gaps": state.get("gaps"),
        "round": (state.get("chat_rounds") or 0) + 1,
    }
    return f"""You are CreditLens Gap Agent — a credit analyst AI for freelancer loans.

Goals:
1. Ask ONE focused question at a time to close information gaps.
2. Acknowledge prior answers briefly when present, then ask the next gap.
3. Never invent scores or approve/deny loans.
4. Keep replies under 120 words.

Context JSON:
{json.dumps(ctx, indent=2)}
"""


def _offline_chat_question(state: CaseState) -> str:
    gaps = state.get("gaps") or ["Confirm loan purpose and amount"]
    rounds = int(state.get("chat_rounds") or 0)
    idx = min(rounds, len(gaps) - 1)
    name = (state.get("applicant") or {}).get("full_name", "there")
    if rounds == 0:
        return (
            f"Hi {name}, I'm your CreditLens analyst agent. "
            f"I've reviewed cashflow, documents, and reputation. "
            f"First question: {gaps[idx]}"
        )
    return f"Thanks — noted. Next: {gaps[idx]}"


def ask_applicant_node(state: CaseState) -> dict[str, Any]:
    """Produce the next analyst question (no interrupt here)."""
    llm = get_llm(temperature=0.4)
    messages = list(state.get("messages") or [])

    if llm is None:
        question = _offline_chat_question(state)
    else:
        lc_messages = [SystemMessage(content=_chat_system_prompt(state)), *messages]
        if not any(isinstance(m, HumanMessage) for m in messages):
            lc_messages.append(
                HumanMessage(content="Start the assessment with the first gap question.")
            )
        try:
            question = str(llm.invoke(lc_messages).content).strip()
        except Exception:  # noqa: BLE001
            question = _offline_chat_question(state)

    return {
        "messages": [AIMessage(content=question)],
        "awaiting_human": True,
        "status": "needs_applicant_input",
    }


def wait_applicant_node(state: CaseState) -> dict[str, Any]:
    """HITL interrupt — resumes with applicant reply."""
    last_ai = ""
    for m in reversed(state.get("messages") or []):
        if isinstance(m, AIMessage) or getattr(m, "type", None) == "ai":
            last_ai = str(m.content)
            break

    answer = interrupt(
        {
            "type": "applicant_question",
            "question": last_ai,
            "case_id": state.get("case_id"),
            "round": int(state.get("chat_rounds") or 0) + 1,
            "max_rounds": state.get("max_chat_rounds") or 3,
            "gaps": state.get("gaps"),
        }
    )

    if isinstance(answer, dict):
        reply = str(answer.get("text") or answer.get("human_input") or "")
    else:
        reply = str(answer)

    return {
        "messages": [HumanMessage(content=reply)],
        "chat_rounds": int(state.get("chat_rounds") or 0) + 1,
        "awaiting_human": False,
        "status": "running",
    }


def should_continue_chat(state: CaseState) -> Literal["ask", "score"]:
    rounds = int(state.get("chat_rounds") or 0)
    max_rounds = int(state.get("max_chat_rounds") or 3)
    return "ask" if rounds < max_rounds else "score"


def score_node(state: CaseState) -> dict[str, Any]:
    scores = compute_potential_score(
        state.get("cashflow") or {},
        state.get("document_analysis") or {},
        state.get("reputation") or {},
    )
    return {"scores": scores}


def explain_node(state: CaseState) -> dict[str, Any]:
    chat_bits = []
    for m in state.get("messages") or []:
        role = getattr(m, "type", None) or m.__class__.__name__
        content = getattr(m, "content", str(m))
        chat_bits.append(f"{role}: {content}")
    applicant_report, analyst_report = generate_reports(
        state.get("applicant") or {},
        state.get("cashflow") or {},
        state.get("document_analysis") or {},
        state.get("reputation") or {},
        state.get("scores") or {},
        chat_summary="\n".join(chat_bits[-12:]),
    )
    return {
        "applicant_report": applicant_report,
        "analyst_report": analyst_report,
        "status": "awaiting_officer",
    }


def officer_review_node(state: CaseState) -> dict[str, Any]:
    decision = interrupt(
        {
            "type": "officer_review",
            "case_id": state.get("case_id"),
            "scores": state.get("scores"),
            "applicant_report": state.get("applicant_report"),
            "analyst_report": state.get("analyst_report"),
            "applicant": state.get("applicant"),
        }
    )

    if isinstance(decision, dict):
        verdict = decision.get("decision") or decision.get("officer_decision") or "request_more_info"
        notes = decision.get("notes") or decision.get("officer_notes") or ""
    else:
        verdict, notes = str(decision), ""

    if verdict not in {"approve", "decline", "request_more_info"}:
        verdict = "request_more_info"

    return {
        "officer_decision": verdict,
        "officer_notes": notes,
        "status": "completed",
    }
