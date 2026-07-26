from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver


class GraphState(TypedDict):
    draft: str
    tool_to_call: str
    tool_args: dict
    tool_approved: bool


def supervisor_agent(state: GraphState) -> GraphState:
    print("🤖 Supervisor: I have decided that the article is ready to be published.")

    return {
        "tool_to_call": "post_on_blog",
        "tool_args": {
            "title": "LangGraph HITL",
            "content": state.get("draft")
        },
        "tool_approved": False  # By default, nothing is approved
    }

def execute_tool(state: GraphState) -> GraphState:
    if not state.get("tool_approved"):
        print("❌ Action rejected. The tool was not authorized.")
        return {"draft": state["draft"] + " [Publication cancelled]"}

    print(f"🚀 Calling tool {state['tool_to_call']} with parameters: {state['tool_args']}")

    return {"draft": state["draft"] + " [Successfully published!]"}


def create_graph():
    graph = StateGraph(GraphState)

    graph.add_node("agent", supervisor_agent)
    graph.add_node("execute_tool", execute_tool)

    graph.add_edge(START, "agent")
    graph.add_edge("agent", "execute_tool")
    graph.add_edge("execute_tool", END)

    return graph.compile(checkpointer=MemorySaver(), interrupt_before=["execute_tool"])


def main():
    graph = create_graph()
    config = {
        "configurable": {
            "thread_id": "auth_test_1"
        }
    }
    initial_state = {
        "draft": "Borrador final de arquitectura.",
        "tool_approved": False
    }

    print("\n--- 1. THE AGENT PROPOSES AN ACTION ---")
    for event in graph.stream(initial_state, config):
        print(event)

    graph_state = graph.get_state(config)
    print(f"\n⚠️ AUTHORIZATION REQUIRED ⚠️")
    print(f"The agent wants to execute: {graph_state.values.get('tool_to_call')}")
    print(f"With this data: {graph_state.values.get('tool_args')}")

    print("\n--- 2. HUMAN DECISION ---")
    can_execute_tool = input("\nAuthorize execution? (y/n): ")

    if can_execute_tool.lower() == 'y':
        print("\n✅ Human: Authorized. Updating status and resuming...")
        # Edit state to approve calling the tool
        graph.update_state(config, {"tool_approved": True}, as_node="agent")
    else:
        print("\n🚫 Human: Denied.  Updating status and resuming...")
        graph.update_state(config, {"tool_approved": False}, as_node="agent")

    print("\n--- 3. RESUME GRAPH ---")
    for event in graph.stream(None, config):
            print(event)

    final_state = graph.get_state(config)
    print("\nFinal state of the article:", final_state.values.get("draft"))


if __name__ == "__main__":
    main()
