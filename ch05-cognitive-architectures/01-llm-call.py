# A larger application that makes use of an LLM for achieving a specific task.
# Some examples using LLM Call Architecture:
# * AI-powered features such as summarize and translate
# * Simple SQL query generation
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama

from langchain_core.messages import HumanMessage

model = ChatOllama(model="gemma4:e2b")

class State(TypedDict):
    # Messages have the type "list". The `add_messages`
    # function in the annotation defines how this state should
    # be updated (in this case, it appends new messages to the
    # list, rather than replacing the previous messages)
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}

builder = StateGraph(State)
builder.add_node('chatbot', chatbot)
builder.add_edge(START, 'chatbot')
builder.add_edge('chatbot', END)

graph = builder.compile()

# Draw a visual representation of the graph
graph_png = graph.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(graph_png)

# You can run it with the stream() method
input = {"messages": [HumanMessage('hi!')]}
for chunk in graph.stream(input):
    print(chunk)
