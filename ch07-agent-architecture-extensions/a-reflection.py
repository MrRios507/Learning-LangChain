# Reflection (also known as self-critique)

# Reflection is the creation of a loop between a creator prompt and a reviser prompt.

# Reflection can be combined with other techniques, such as chain-of-thought and tool calling.

# We'll implement reflection as a graph with two nodes: generate and reflect.
# This graph will be tasked with writing three-paragraph essay
# - The generate node: writing or revising drafts of the essay.
# - The reflect node: writing a critique to inform the next revision.

# We'll run the loop a fixed number of times, but a variation on this technique would be to have the reflect node decide when to finish.

from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

model = ChatOllama(model="gemma2:2b")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


generate_prompt = SystemMessage(
    """You are an essay assistant tasked with writing excellent 3-paragraph essays."""
    "Generate the best essay possible for the user's request."
    """If the user provides critique, respond with a revised version of your previous attempts."""
)

def generate(state: State) -> State:
    answer = model.invoke([generate_prompt] + state["messages"])
    return {"messages": [answer]}

reflection_prompt = SystemMessage(
    """You are a teacher grading an essay submission. Generate critique and recommendations for the use's submission."""
    """Provide detailed recommendations, including requests for length, depth, styel, etc."""
)

def reflect(state: State) -> State:
    # Invert the messages to get the LLM to reflect on its own output
    cls_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
    # First message is the original user request.
    # We hold it the same for all nodes
    translated = [reflection_prompt, state["messages"][0]] + [
        cls_map[msg.__class__](content=msg.content) for msg in state["messages"][1:]
    ]
    answer = model.invoke(translated)
    # We treat the output of this as human feedbak for the generator
    return {"messages": HumanMessage(content=answer.content)}

def should_continue(state: State):
    if len(state["messages"]) > 6:
        # End after 3 iterations, each with 2 messages
        return END
    else:
        return "reflect"

builder = StateGraph(State)

builder.add_node("generate", generate)
builder.add_node("reflect", reflect)

builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")

graph = builder.compile()

# Draw a visual representation of the graph
graph_png = graph.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(graph_png)


# Example usage
initial_state = {
    "messages": [
        HumanMessage(
            content="Write an essay about the relevance of 'The Little Prince' today."
        )
    ]
}

# Run the graph
for c in graph.stream(initial_state):
    print(c)
    print(110*"=")