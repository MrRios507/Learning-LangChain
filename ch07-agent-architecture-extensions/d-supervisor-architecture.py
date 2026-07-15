# In this architecture, we add each agent to the graph as a node and also a supervisor node.
# - The supervisor node decides which agents should be called next.
# - We use conditional edges to route execution to the appropriate agent node based on the supervisor's decision.

from typing import Literal

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from pydantic import BaseModel


# -------------------------
# Supervisor output schema
# -------------------------
class SupervisorDecision(BaseModel):
    next: Literal["researcher", "coder", "FINISH"]


# -------------------------
# State
# -------------------------
class AgentState(MessagesState):
    next: Literal["researcher", "coder", "FINISH"] | None = None


# -------------------------
# Models
# -------------------------
llm = ChatOllama(
    model="granite4.1:3b",
    temperature=0,
)

supervisor_llm = llm.with_structured_output(SupervisorDecision)


# -------------------------
# Prompt
# -------------------------
agents = ["researcher", "coder"]

system_prompt_part_1 = f"""
You are a supervisor managing a conversation between these workers:
{agents}.

Given the user's request, decide which worker should act next.

Each worker performs its task and appends its response to the conversation.

When the task is complete respond with FINISH.
"""

system_prompt_part_2 = f"""
Based on the conversation, who should act next?

Choose only one of:

{", ".join(agents)}, FINISH
"""


# -------------------------
# Supervisor
# -------------------------
def supervisor(state: AgentState):

    messages = [
        ("system", system_prompt_part_1),
        *state["messages"],
        ("system", system_prompt_part_2),
    ]

    decision = supervisor_llm.invoke(messages)

    print(f"\nSupervisor -> {decision.next}")

    return {
        "next": decision.next
    }


# -------------------------
# Researcher
# -------------------------
def researcher(state: AgentState):

    response = llm.invoke(
        [
            (
                "system",
                "You are a research assistant. Analyze the request and provide useful information.",
            ),
            *state["messages"],
        ]
    )

    return {
        "messages": [response]
    }


# -------------------------
# Coder
# -------------------------
def coder(state: AgentState):

    response = llm.invoke(
        [
            (
                "system",
                "You are a software engineer. Write code or technical solutions when needed.",
            ),
            *state["messages"],
        ]
    )

    return {
        "messages": [response]
    }


# -------------------------
# Graph
# -------------------------
builder = StateGraph(AgentState)

builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("coder", coder)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "researcher": "researcher",
        "coder": "coder",
        "FINISH": END,
    },
)

builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")

graph = builder.compile()


# -------------------------
# Run
# -------------------------
initial_state = {
    "messages": [
        {
            "role": "user",
            "content": "I need help analyzing some data and creating a visualization.",
        }
    ]
}

for step in graph.stream(initial_state):
    print(step)
    print("=" * 100)