from __future__ import annotations

import uuid
from typing import Any, Optional

from langgraph.types import Command

from creditlens_agentic.graph import get_compiled_graph
from creditlens_agentic.tools import list_personas


def _serialize_messages(messages: list) -> list[dict[str, str]]:
    out = []
    for m in messages or []:
        role = getattr(m, "type", None)
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        else:
            role = m.__class__.__name__.replace("Message", "").lower() or "assistant"
        out.append({"role": role, "content": str(getattr(m, "content", m))})
    return out


def _public_state(values: dict[str, Any], interrupted: Any = None) -> dict[str, Any]:
    return {
        "case_id": values.get("case_id"),
        "persona_id": values.get("persona_id"),
        "applicant": values.get("applicant"),
        "status": values.get("status"),
        "cashflow": values.get("cashflow"),
        "document_analysis": values.get("document_analysis"),
        "reputation": values.get("reputation"),
        "gaps": values.get("gaps"),
        "scores": values.get("scores"),
        "applicant_report": values.get("applicant_report"),
        "analyst_report": values.get("analyst_report"),
        "officer_decision": values.get("officer_decision"),
        "officer_notes": values.get("officer_notes"),
        "chat_rounds": values.get("chat_rounds"),
        "max_chat_rounds": values.get("max_chat_rounds"),
        "messages": _serialize_messages(values.get("messages") or []),
        "interrupt": interrupted,
    }


def start_case(persona_id: str = "persona1", max_chat_rounds: int = 3) -> dict[str, Any]:
    if persona_id not in list_personas():
        raise ValueError(f"Unknown persona '{persona_id}'. Available: {list_personas()}")

    graph = get_compiled_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        {"persona_id": persona_id, "max_chat_rounds": max_chat_rounds},
        config=config,
    )

    # After first interrupt, result may be a state dict; check tasks via get_state
    snap = graph.get_state(config)
    interrupted = None
    if snap.tasks:
        for task in snap.tasks:
            if task.interrupts:
                interrupted = task.interrupts[0].value
                break

    values = snap.values if snap.values else result
    public = _public_state(values, interrupted)
    public["thread_id"] = thread_id
    return public


def resume_case(thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}}

    graph.invoke(Command(resume=payload), config=config)
    snap = graph.get_state(config)

    interrupted = None
    if snap.tasks:
        for task in snap.tasks:
            if task.interrupts:
                interrupted = task.interrupts[0].value
                break

    public = _public_state(snap.values, interrupted)
    public["thread_id"] = thread_id
    return public


def get_case(thread_id: str) -> Optional[dict[str, Any]]:
    graph = get_compiled_graph()
    config = {"configurable": {"thread_id": thread_id}}
    snap = graph.get_state(config)
    if not snap or not snap.values:
        return None

    interrupted = None
    if snap.tasks:
        for task in snap.tasks:
            if task.interrupts:
                interrupted = task.interrupts[0].value
                break

    public = _public_state(snap.values, interrupted)
    public["thread_id"] = thread_id
    return public
