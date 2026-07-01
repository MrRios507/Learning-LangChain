# This architecture extends on all that by using multiple LLM calls, in a predefined sequence.
# An example a text-to-SQL application, which receives as input from the user a natural language description
# 1. One LLM call to generate a SQL query from the natural language query, provided by the user.
# 2. Another LLM call to write an explanation of the query appropriate for a nontechnical user

from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# useful to generate SQL query
model_low_temp = ChatOllama(model="gemma4:e2b", temperature=0.1)
# useful to generate natural language outputs
model_high_temp = ChatOllama(model="gemma4:e2b", temperature=0.7)

class State(TypedDict):
    # to track conversation history
    messages: Annotated[list, add_messages]
    # input
    user_query: str
    # output
    sql_query: str
    sql_explanation: str

class Input(TypedDict):
    user_query: str

class Output(TypedDict):
    sql_query: str
    sql_explanation: str

generate_prompt = SystemMessage(
    """You are a helpful data analyst who generates SQL queries for users based on their questions."""
)

def generate_sql(state: State) -> State:
    user_message = HumanMessage(state["user_query"])
    messages = [generate_prompt, *state["messages"], user_message]
    res = model_low_temp.invoke(messages)
    
    return {
        "sql_query": res.content,
        # update conversation history
        "messages": [user_message, res]
    }

explain_prompt = SystemMessage(
    """You are a helpful data analyst who explains SQL queries to users."""
)

def explain_sql(state: State) -> State:
    messages = [
        explain_prompt,
        # contains user's query and SQL query from prev step
        *state["messages"],
    ]
    res = model_high_temp.invoke(messages)

    return {
        "sql_explanation": res.content,
        # update conversation history
        "messages": res,
    }

builder = StateGraph(State, input_schema=Input, output_schema=Output)
builder.add_node("generate_sql", generate_sql)
builder.add_node("explain_sql", explain_sql)
builder.add_edge(START, "generate_sql")
builder.add_edge("generate_sql", "explain_sql")
builder.add_edge("explain_sql", END)

graph = builder.compile()

result = graph.invoke({
    "user_query": "What is the total sales for each product?"
})
print(result)
