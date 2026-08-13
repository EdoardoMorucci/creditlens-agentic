from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class CaseState(TypedDict, total=False):
    """Shared state for one loan application case."""

    case_id: str
    persona_id: str
    applicant: dict[str, Any]

    # Ingested raw sources
    banking: dict[str, Any]
    invoices: dict[str, Any]
    payments: dict[str, Any]
    reputation_raw: dict[str, Any]
    documents_text: str

    # Deterministic / LLM analysis outputs
    cashflow: dict[str, Any]
    document_analysis: dict[str, Any]
    reputation: dict[str, Any]

    # Gap / chat
    gaps: list[str]
    messages: Annotated[list, add_messages]
    chat_rounds: int
    max_chat_rounds: int
    awaiting_human: bool
    human_input: Optional[str]

    # Scoring (deterministic) + narrative
    scores: dict[str, Any]
    applicant_report: str
    analyst_report: str

    # Officer HITL
    officer_decision: Optional[Literal["approve", "decline", "request_more_info"]]
    officer_notes: Optional[str]
    status: Literal[
        "running",
        "needs_applicant_input",
        "awaiting_officer",
        "completed",
        "error",
    ]
    error: Optional[str]
