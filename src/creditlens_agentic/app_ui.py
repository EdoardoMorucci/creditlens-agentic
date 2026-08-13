from __future__ import annotations

import streamlit as st

from creditlens_agentic.service import get_case, resume_case, start_case
from creditlens_agentic.tools import list_personas


def main() -> None:
    st.set_page_config(page_title="CreditLens Agentic", page_icon="💠", layout="wide")
    st.title("CreditLens Agentic")
    st.caption("Same credit signals as CreditLens — LangGraph agents, deterministic scoring, HITL.")

    personas = list_personas() or ["persona1"]

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
        st.session_state.case = None

    with st.sidebar:
        st.header("New case")
        persona = st.selectbox("Persona", personas)
        rounds = st.slider("Chat rounds", 1, 5, 3)
        if st.button("Start assessment", type="primary"):
            with st.spinner("Running ingest + analysis…"):
                case = start_case(persona, rounds)
            st.session_state.thread_id = case["thread_id"]
            st.session_state.case = case
            st.rerun()

        if st.session_state.thread_id:
            st.divider()
            st.code(st.session_state.thread_id, language=None)
            if st.button("Refresh state"):
                st.session_state.case = get_case(st.session_state.thread_id)
                st.rerun()

    case = st.session_state.case
    if not case:
        st.info("Start a case from the sidebar. Works offline without API keys (fallback mode).")
        return

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Applicant")
        st.write(case.get("applicant") or {})
        st.subheader("Gaps")
        for g in case.get("gaps") or []:
            st.markdown(f"- {g}")

        st.subheader("Conversation")
        for m in case.get("messages") or []:
            with st.chat_message("user" if m["role"] == "user" else "assistant"):
                st.markdown(m["content"])

        interrupt = case.get("interrupt") or {}
        itype = interrupt.get("type") if isinstance(interrupt, dict) else None

        if itype == "applicant_question":
            st.markdown("**Agent is waiting for your answer**")
            st.info(interrupt.get("question", ""))
            reply = st.chat_input("Your reply")
            if reply:
                with st.spinner("Resuming graph…"):
                    st.session_state.case = resume_case(
                        st.session_state.thread_id, {"text": reply}
                    )
                st.rerun()

        elif itype == "officer_review":
            st.markdown("**Loan officer review (HITL)**")
            decision = st.radio(
                "Decision",
                ["approve", "decline", "request_more_info"],
                horizontal=True,
            )
            notes = st.text_area("Notes")
            if st.button("Submit decision"):
                with st.spinner("Closing case…"):
                    st.session_state.case = resume_case(
                        st.session_state.thread_id,
                        {"decision": decision, "notes": notes},
                    )
                st.rerun()

        elif case.get("status") == "completed":
            st.success(
                f"Completed — decision: **{case.get('officer_decision')}** "
                f"({case.get('officer_notes') or 'no notes'})"
            )

    with right:
        scores = case.get("scores") or {}
        if scores:
            st.subheader("Potential Score (deterministic)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("General", scores.get("general_score"))
            c2.metric("Financial", scores.get("financial_health_score"))
            c3.metric("Future income", scores.get("future_income_score"))
            c4.metric("Reputation", scores.get("reputation_score"))
            st.write(
                f"**{scores.get('recommendation')}** · risk {scores.get('risk_level')} · "
                f"{scores.get('max_loan_amount')}"
            )

        with st.expander("Cashflow analysis", expanded=False):
            st.json(case.get("cashflow") or {})
        with st.expander("Document analysis", expanded=False):
            st.json(case.get("document_analysis") or {})
        with st.expander("Reputation", expanded=False):
            st.json(case.get("reputation") or {})

        if case.get("applicant_report"):
            st.subheader("Applicant report")
            st.markdown(case["applicant_report"])
        if case.get("analyst_report"):
            st.subheader("Analyst report")
            st.markdown(case["analyst_report"])


if __name__ == "__main__":
    main()
