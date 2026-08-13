# CreditLens Agentic

Stessi segnali di **CreditLens** (cashflow Open Banking, documenti/fatture, reputazione marketplace, report applicant/analyst), ma con controllo **agentico** via **LangGraph**.

## Cosa è agentico (e cosa no)

| Pezzo | Comportamento |
|-------|----------------|
| Ingest + cashflow | Tool deterministici (pandas) |
| Document / reputation analysis | Tool LLM structured (fallback offline) |
| Gap agent + chat | Agente conversazionale con loop e **HITL** (`interrupt`) |
| Potential Score | **Sempre deterministico** (l'LLM non inventa il voto) |
| Explain | Narrativa LLM (o template offline) |
| Officer review | **HITL obbligatorio** prima della chiusura |

```text
ingest → analyze_all → gap → ask ⇄ wait(HITL) → score → explain → officer(HITL) → END
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
copy .env.example .env   # oppure: cp .env.example .env
```

Senza API key funziona in **offline mode** (fallback deterministici + chat template).

Con LLM:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
# oppure
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
```

### UI Streamlit

```bash
streamlit run src/creditlens_agentic/app_ui.py
```

### API FastAPI

```bash
uvicorn creditlens_agentic.api:app --reload --app-dir src
```

Esempio:

```bash
curl -X POST http://localhost:8000/cases -H "Content-Type: application/json" -d "{\"persona_id\":\"persona1\",\"max_chat_rounds\":2}"
curl -X POST http://localhost:8000/cases/<thread_id>/reply -H "Content-Type: application/json" -d "{\"text\":\"Serve un prestito da 8k per attrezzatura\"}"
curl -X POST http://localhost:8000/cases/<thread_id>/officer -H "Content-Type: application/json" -d "{\"decision\":\"approve\",\"notes\":\"ok\"}"
```

### Test

```bash
pytest -q
```

## Differenza vs CreditLens originale

- Orchestrazione: pipeline Lambda fissa → **grafo LangGraph** con checkpoint in-memory
- Chat: dump nel system prompt → **agente gap-filling** con pause/resume
- Score: mix LLM + formule → **solo formule** (AI Act / auditabilità)
- Officer: assente → **interrupt HITL**

## Layout

```text
src/creditlens_agentic/
  graph/          # LangGraph nodes + builder
  tools/          # cashflow, docs, reputation, scoring, gaps/reports
  api.py          # FastAPI
  app_ui.py       # Streamlit demo
  service.py      # start/resume case
data/personas/    # mock data (persona1 da CreditLens)
```

## Team / origine

Riscrittura agentica del concept CreditLens (AI & Financial Services Hackathon 2025).
