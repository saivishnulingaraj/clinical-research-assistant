"""Assemble the research -> summarize -> respond pipeline as a LangGraph.

The graph is linear by design (research feeds summarize feeds respond). Using
LangGraph gives us a typed, inspectable state machine and an easy path to add
branches later (e.g. a re-retrieval loop) without restructuring the agents.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from cra.agents.context import AgentContext
from cra.agents.research import research_node
from cra.agents.respond import respond_node
from cra.agents.summarize import summarize_node
from cra.graph.state import GraphState


def build_graph(ctx: AgentContext):
    """Compile the three-agent graph bound to a per-request context."""
    graph = StateGraph(GraphState)

    graph.add_node("research", partial(research_node, ctx=ctx))
    graph.add_node("summarize", partial(summarize_node, ctx=ctx))
    graph.add_node("respond", partial(respond_node, ctx=ctx))

    graph.set_entry_point("research")
    graph.add_edge("research", "summarize")
    graph.add_edge("summarize", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


def run_pipeline(state: GraphState, ctx: AgentContext) -> GraphState:
    """Run the compiled graph and return the final typed state.

    LangGraph may hand node outputs back as dicts; we coerce the terminal state
    back into :class:`GraphState` so callers always get the typed contract.
    """
    app = build_graph(ctx)
    result = app.invoke(state)
    if isinstance(result, GraphState):
        return result
    return GraphState.model_validate(result)
