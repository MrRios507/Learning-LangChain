# Define a subgraph with a completely different schema.
# - Create a node with a function that invokes the subgraph. This function will need to transform the input (parent) state
#   to the subgraph state before invoking the subgraph and transform the results back to the parent state before returning the state update from the node.
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

class State(TypedDict):
    foo: str

class SubgraphState(TypedDict):
    # none of these keys are shared with the parent graph state
    bar: str
    baz: str

# Define subgraph
def subgraph_node(state: SubgraphState) -> SubgraphState:
    return {"bar": state["bar"] + "baz"}

subgraph_builder = StateGraph(SubgraphState)

subgraph_builder.add_node("subgraph_node", subgraph_node)

subgraph_builder.add_edge(START, "subgraph_node")
subgraph_builder.add_edge("subgraph_node", END)

subgraph = subgraph_builder.compile()


# Define parent graph
def node(state: State) -> State:
    # transform the state to the subgraph state
    response = subgraph.invoke({"bar": state["foo"]})
    # transform response back to the parent state
    return {"foo": response["bar"]}

builder = StateGraph(State)
# note that we are using `node` function instead of a compiled subgraph
builder.add_node("node", node)

builder.add_edge(START, "node")
builder.add_edge("node", END)
# Additional parent graph setup would go here
graph = builder.compile()

# Example usage
initial_state = {"foo": "hello"}
result = graph.invoke(initial_state)
print(
    f"Result: {result}"
)