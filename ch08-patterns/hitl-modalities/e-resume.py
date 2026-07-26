# To proceed from an interrupted graph, such as when using interrupt or authorize, 
# you just need to re-invoke the graph with null input (or None in Python).
import asyncio
from contextlib import aclosing

from langchain.messages import HumanMessage

from graph import create_graph


async def main():
    graph = create_graph()

    config = {"configurable": {"thread_id": "1"}}

    input = {
        "messages": [
            HumanMessage(
                "How old was the 30th president of the United States when he died?"
            )
        ]
    }

    output = graph.astream(input, config, interrupt_before=["tools"])

    print("\n--- 1. START GRAPH ---")

    async for c in graph.astream(input, config, interrupt_before=["tools"]):
        print(c)

    graph_state = await graph.aget_state(config)
    print(f"Current status: paused. Next node: {graph_state.next}")

    print("\n--- 2. RESUME GRAPH ---")
    # Re-invoke the node with None
    async for c in graph.astream(None, config):
        print(c)

    print("Final status: ", graph.get_state(config).values)


if __name__ == "__main__":
    asyncio.run(main())