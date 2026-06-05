from typing import Annotated, TypedDict

from langchain.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_ollama import ChatOllama

class State(TypedDict):
    # Messages have the type "list". The `add_messages`
    # function in the annotation defines how this state should
    # be updated (in this case, it appends new messages to the
    # list, rather than replacing the previous messages)
    messages: Annotated[list, add_messages]

builder = StateGraph(State)

model = ChatOllama(model="gemma2:2b")

def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


# The first argument is the unique node name
# The second argument is the function or Runnable to run
builder.add_node("chatbot", chatbot)

# Add the edges
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# Draw a visual representation of the graph
# graph.get_graph().draw_mermaid_png()

# Run the graph
input = {"messages": [HumanMessage("hi!")]}

for chunk in graph.stream(input):
    print(chunk)

