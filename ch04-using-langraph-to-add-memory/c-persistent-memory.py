from typing import Annotated, TypedDict

from langchain.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

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

# Add persistence with MemorySaver
graph = builder.compile(checkpointer=MemorySaver())

thread1 = {"configurable": {"thread_id": "1"}} # Identify the current interaction
result_1 = graph.invoke(
    {"messages": [HumanMessage("hi, my name is Jack!")]},
    thread1
)

print(result_1)

result_2 = graph.invoke(
    {"messages": [HumanMessage("what is my name?")]},
    thread1
)

print(result_2)

# Get the state of the graph
print(graph.get_state(thread1))