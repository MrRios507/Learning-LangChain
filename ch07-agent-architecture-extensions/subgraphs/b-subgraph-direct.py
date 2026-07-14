# The simplest way to create a subgraph nodes is to attach a subgraph directly as a node.

# It is important that the parent graph and the subgraph share state keys, because those shared keys will be used to communicate.

# If you pass extra keys to the subgraph node (that is, in addition to the shared keys), they will ignore by the subgraph node.
# - Similary, if you return extra keys from the subgraph, they will be ignored by the parent graph.

from langgraph.graph import END, START, StateGraph
from typing import TypedDict

class State(TypedDict):
    foo: str    # this key is shared with the subgraph


class SubgraphState(TypedDict):
    foo: str    # this key is shared with the parent graph
    bar: str

# Define subgraph
def subgraph_node(state: SubgraphState):
    # note that this subgraph node can communicate with the parent graph
    # via the shared "foo" key
    return {"foo": state["foo"] + "bar"}


subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("subgraph_node", subgraph_node)

subgraph_builder.add_edge(START, "subgraph_node")
subgraph_builder.add_edge("subgraph_node", END)

# Additional subgraph setup would go here
subgraph = subgraph_builder.compile()

# Define parent graph
builder = StateGraph(State)
builder.add_node("subgraph", subgraph)

builder.add_edge(START, "subgraph")
builder.add_edge("subgraph", END)

# Additional parent graph setup would go here
graph = builder.compile()

# Example usage
initial_state = {"foo": "hello"}
result = graph.invoke(initial_state)
print(f"Result: {result}")  # Should append "bar" to the foo value
