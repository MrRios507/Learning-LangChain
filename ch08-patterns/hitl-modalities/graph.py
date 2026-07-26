import ast
from typing import Annotated, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver
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

def create_graph():
    builder = StateGraph(State)

    builder.add_node("select_tools", select_tools)
    builder.add_node("model", model_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "select_tools")
    builder.add_edge("select_tools", "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")

    return builder.compile(checkpointer=MemorySaver())
