from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from creditlens_agentic.graph.nodes import (
    analyze_all_node,
    ask_applicant_node,
    explain_node,
    gap_node,
    ingest_node,
    officer_review_node,
    score_node,
    should_continue_chat,
    wait_applicant_node,
)
from creditlens_agentic.state import CaseState


def build_graph(checkpointer: MemorySaver | None = None):
    """
    Agentic credit assessment graph:

      ingest → analyze_all → gap → ask ⇄ wait (HITL) → score → explain → officer HITL → END

    Scoring is deterministic. Chat + explain use LLM when keys are set.
    """
    g = StateGraph(CaseState)

    g.add_node("ingest", ingest_node)
    g.add_node("analyze_all", analyze_all_node)
    g.add_node("gap", gap_node)
    g.add_node("ask", ask_applicant_node)
    g.add_node("wait_applicant", wait_applicant_node)
    g.add_node("score", score_node)
    g.add_node("explain", explain_node)
    g.add_node("officer_review", officer_review_node)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "analyze_all")
    g.add_edge("analyze_all", "gap")
    g.add_edge("gap", "ask")
    g.add_edge("ask", "wait_applicant")
    g.add_conditional_edges(
        "wait_applicant",
        should_continue_chat,
        {"ask": "ask", "score": "score"},
    )
    g.add_edge("score", "explain")
    g.add_edge("explain", "officer_review")
    g.add_edge("officer_review", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


@lru_cache
def get_compiled_graph():
    return build_graph()
