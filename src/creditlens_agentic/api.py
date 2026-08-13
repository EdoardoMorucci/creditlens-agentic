from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from creditlens_agentic.service import get_case, resume_case, start_case
from creditlens_agentic.tools import list_personas

app = FastAPI(
    title="CreditLens Agentic",
    description="LangGraph credit assessment — same signals as CreditLens, agentic control flow",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartRequest(BaseModel):
    persona_id: str = "persona1"
    max_chat_rounds: int = Field(default=3, ge=1, le=8)


class ApplicantReply(BaseModel):
    text: str


class OfficerDecision(BaseModel):
    decision: Literal["approve", "decline", "request_more_info"]
    notes: str = ""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/personas")
def personas() -> dict[str, list[str]]:
    return {"personas": list_personas()}


@app.post("/cases")
def create_case(body: StartRequest) -> dict[str, Any]:
    try:
        return start_case(body.persona_id, body.max_chat_rounds)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/cases/{thread_id}")
def read_case(thread_id: str) -> dict[str, Any]:
    case = get_case(thread_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@app.post("/cases/{thread_id}/reply")
def reply(thread_id: str, body: ApplicantReply) -> dict[str, Any]:
    try:
        return resume_case(thread_id, {"text": body.text})
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/cases/{thread_id}/officer")
def officer(thread_id: str, body: OfficerDecision) -> dict[str, Any]:
    try:
        return resume_case(
            thread_id,
            {"decision": body.decision, "notes": body.notes},
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    import uvicorn

    uvicorn.run(
        "creditlens_agentic.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
