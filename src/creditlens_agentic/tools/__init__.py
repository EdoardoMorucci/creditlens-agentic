from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from creditlens_agentic.config import get_settings


def list_personas() -> list[str]:
    root = get_settings().data_path
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_persona(persona_id: str) -> dict[str, Any]:
    """Load mock banking / invoices / payments / reputation for a persona."""
    base = get_settings().data_path / persona_id
    if not base.exists():
        raise FileNotFoundError(f"Persona not found: {persona_id} ({base})")

    # Support both persona1-banking.json and banking.json naming
    def pick(*names: str) -> dict[str, Any]:
        for name in names:
            data = _load_json(base / name)
            if data:
                return data
        return {}

    banking = pick(f"{persona_id}-banking.json", "banking.json")
    invoices = pick(f"{persona_id}-invoices.json", "invoices.json")
    payments = pick(f"{persona_id}-payments.json", "payments.json")
    reputation = pick(f"{persona_id}-reputation.json", "reputation.json")

    applicant = {
        "full_name": reputation.get("name")
        or (banking.get("accounts") or [{}])[0].get("owner_name", "Unknown"),
        "headline": reputation.get("headline", ""),
        "skills": reputation.get("skills", []),
        "persona_id": persona_id,
    }

    docs_parts: list[str] = []
    if invoices:
        docs_parts.append("## Invoices\n" + json.dumps(invoices, indent=2))
    if payments:
        docs_parts.append("## Payments\n" + json.dumps(payments, indent=2))

    return {
        "applicant": applicant,
        "banking": banking,
        "invoices": invoices,
        "payments": payments,
        "reputation_raw": reputation,
        "documents_text": "\n\n".join(docs_parts),
    }
