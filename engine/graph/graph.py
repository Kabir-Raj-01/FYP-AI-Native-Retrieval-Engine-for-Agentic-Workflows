"""The LangGraph graph of one controlled iteration (ARCHITECTURE §4; ADR-002, ADR-009 §2).

``maybe_plan -> select_action -> execute -> score_and_update -> monitor -> persist -> END``.
The graph runs exactly one iteration per invoke; ``Engine.run`` loops over iterations
itself (between iterations it checks ``api.run_timeout_s`` and ``status``), so the graph
has no loop edge and no LangGraph checkpointer — TaskStore is the source of truth.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from engine.graph.nodes import (
    Context,
    execute,
    maybe_plan,
    monitor,
    persist,
    score_and_update,
    select_action,
)
from engine.state.models import GraphState, RetrievalState

__all__ = ["NODES", "run_iteration"]

# each node is node(gs, ctx=...) -> partial update (engine/graph/nodes.py)
NODES: tuple[Callable[..., dict[str, Any]], ...] = (
    maybe_plan,
    select_action,
    execute,
    score_and_update,
    monitor,
    persist,
)


def run_iteration(state: RetrievalState, ctx: Context) -> RetrievalState:
    """Run one iteration on ``state`` and return the persisted new state."""
    graph = StateGraph(GraphState)
    for node in NODES:
        graph.add_node(node.__name__, partial(node, ctx=ctx))
    for a, b in itertools.pairwise([START, *(n.__name__ for n in NODES), END]):
        graph.add_edge(a, b)
    # recursion_limit is only a safety net: one pass through the nodes plus a margin
    out = graph.compile().invoke(
        GraphState(state=state), config={"recursion_limit": len(NODES) + 1}
    )
    return GraphState.model_validate(out).state  # invoke returns a dict, unvalidated
