# CreditLens Agentic

**AI-powered lending assessment for freelancers — orchestrated with LangGraph.**

CreditLens Agentic evaluates loan applications from freelancers, creators, and gig workers using signals that traditional credit scores miss: bank cashflow, invoice/payment documents, and marketplace reputation. The system produces an explainable **Potential Score**, gathers missing context through a conversational analyst, and requires a human loan officer to approve or decline before the case closes.

Built as an agentic evolution of the CreditLens concept from the **AI & Financial Services Hackathon 2025**.

## What it does

1. Loads an applicant profile (Open Banking–style transactions, invoices, payments, reputation).
2. Runs cashflow, document, and reputation analysis.
3. Detects information gaps and interviews the applicant.
4. Computes a deterministic Potential Score (0–1000).
5. Generates applicant-facing and analyst reports.
6. Pauses for loan-officer review (approve / decline / request more info).

```text
ingest → analyze_all → gap → ask ⇄ wait (applicant HITL)
       → score → explain → officer review (HITL) → END
```

## Agents & roles

| Role | Responsibility |
|------|----------------|
| **Ingest** | Loads persona/mock application data into shared case state. |
| **Analysis suite** | Cashflow scoring (pandas), document analysis (structured LLM), reputation analysis (structured LLM). |
| **Gap Agent** | Reviews analysis outputs and builds a prioritized list of missing information. |
| **Chat Analyst** | Conversational agent that asks one focused question per round to close gaps; pauses until the applicant replies. |
| **Scorer** | Fixed-weight formula for Financial Health, Future Income, and Reputation → Potential Score. |
| **Explainer** | Writes the applicant summary and the credit-analyst memo from case state + chat context. |
| **Loan Officer (HITL)** | Human-in-the-loop gate: final decision with optional notes. |

Shared state is a LangGraph `CaseState` (applicant, analyses, gaps, messages, scores, reports, officer decision), checkpointed so the case can pause and resume across HITL steps.

## Stack & frameworks

| Layer | Technology |
|-------|------------|
| Agent orchestration | **LangGraph** (StateGraph, conditional edges, `interrupt`, MemorySaver checkpoints) |
| LLM interface | **LangChain** + OpenAI / Anthropic chat models |
| Structured outputs | **Pydantic** models for document & reputation analysis |
| Analytics | **pandas** / **NumPy** for cashflow metrics |
| API | **FastAPI** |
| Demo UI | **Streamlit** |
| Config | **pydantic-settings** + `.env` |
| Tests | **pytest** |

LLM providers are selectable via `LLM_PROVIDER` (`openai` or `anthropic`). Without API keys the graph still runs in **offline mode** with deterministic fallbacks for analysis, chat prompts, and reports.

## Potential Score

Final scoring is formula-based (not free-form LLM judgment):

- **Financial health** — 40% (cashflow + document evidence)
- **Future income** — 30% (retention, applicability, income stability)
- **Reputation** — 30% (marketplace profile)

Outputs include risk level, recommendation, suggested loan band, and interest-rate guidance for the officer memo.

## Project layout

```text
src/creditlens_agentic/
  graph/          # LangGraph nodes and graph builder
  tools/          # cashflow, documents, reputation, scoring, reports
  api.py          # FastAPI endpoints
  app_ui.py       # Streamlit demo
  service.py      # start / resume case helpers
  state.py        # shared CaseState
data/personas/    # mock applicants (e.g. persona1)
tests/            # end-to-end HITL flow tests
```

## Quick start

```bash
cd creditlens-agentic
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -e ".[dev]"
cp .env.example .env   # Windows: copy .env.example .env
```

Optional LLM config in `.env`:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# or
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
```

### Streamlit UI

```bash
streamlit run src/creditlens_agentic/app_ui.py
```

### FastAPI

```bash
uvicorn creditlens_agentic.api:app --reload --app-dir src
```

```bash
# Start a case
curl -X POST http://localhost:8000/cases \
  -H "Content-Type: application/json" \
  -d "{\"persona_id\":\"persona1\",\"max_chat_rounds\":2}"

# Applicant reply
curl -X POST http://localhost:8000/cases/<thread_id>/reply \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"I need about €8k for equipment. No other debts.\"}"

# Officer decision
curl -X POST http://localhost:8000/cases/<thread_id>/officer \
  -H "Content-Type: application/json" \
  -d "{\"decision\":\"approve\",\"notes\":\"Strong freelancing profile\"}"
```

### Tests

```bash
pytest -q
```

## Notable design choices

- **Human-in-the-loop by default** — applicant Q&A and officer decision use LangGraph `interrupt`, so cases persist across pauses.
- **Audit-friendly scoring** — the Potential Score is reproducible from inputs; LLMs explain and interview, they do not invent the number.
- **Offline-first demo** — useful for local demos and CI without provider keys.
- **Mock personas** — synthetic freelance profiles under `data/personas/` for end-to-end walks without live Open Banking credentials.

## License

Developed for portfolio and educational use around the CreditLens / AI & FS Hackathon concept. All rights reserved by the author unless otherwise noted.
