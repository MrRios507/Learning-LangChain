# LLMs are far from perfect, and they currently struggle more when given multiple choices or excessive information in a prompt.
# These limitations also extend to the planning of the next action to take. When given many tools (say, more than 10) the planning performance starts to suffer.

# The solution to this problem is to reduce the number of tools the LLM can choose from.
# What if you do have many tools you want to see used for different user queries?

# One elegant solution is to use a RAG step to preselect the most relevant tools for the current query and then feed the LLM only that subset of tools instead of the entire arsenal.

import ast
from typing import Annotated, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def calculator(query: str) -> str:
    """A simple calculator tool. Input should be a mathematical expression."""
    return ast.literal_eval(query)

search = DuckDuckGoSearchRun()

tools = [search, calculator]

embeddings = OllamaEmbeddings(model="embeddinggemma")
model = ChatOllama(model="llama3.2:latest", temperature=0.1)

tools_retriever = InMemoryVectorStore.from_documents(
    [Document(tool.description, metadata={"name": tool.name}) for tool in tools],
    embeddings
).as_retriever()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]

def model_node(state: State) -> State:
    selected_tools = [
        tool for tool in tools if tool.name in state["selected_tools"]
    ]
    res = model.bind_tools(selected_tools).invoke(state["messages"])

    return {"messages": res}

def select_tools(state: State) -> State:
    query = state["messages"][-1].content
    tools_docs = tools_retriever.invoke(query)

    return {"selected_tools": [doc.metadata["name"] for doc in tools_docs]}

builder = StateGraph(State)

builder.add_node("select_tools", select_tools)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "select_tools")
builder.add_edge("select_tools", "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

graph = builder.compile()

# Draw a visual representation of the graph
graph_png = graph.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(graph_png)

input = {
    "messages": [
        HumanMessage("""How old was the 30th president of the United States when he died?""")
    ]
}
# To generate intermediate outputs with LangGraph, all you have
# to do is to invoke the graph with the stream method
for c in graph.stream(input, stream_mode='updates'):
    print(c)
    print(110*"=")

