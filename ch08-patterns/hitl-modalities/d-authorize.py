# The authorize mode, where the user defines ahead of time that they want the application to hand off control
# to every time a particular node is about to be called.

# This is usually implemented for tool confirmation before any tool (or particular tools) is called.
# The application will pause and ask for confirmation, at which point the user can, again:
# - Resume computation, accepting the tool call.
# - Send a new message to guide the bot in different direction.
# - Do nothing.

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

    async for c in output:
        print(c)

    graph_state = await graph.aget_state(config)
    messages = graph_state.values.get('messages', [])

    print(f"\n⚠️ AUTHORIZATION REQUIRED ⚠️")

    if messages:
        last_message = messages[-1]
        
        # 2. Validamos si el último mensaje contiene intenciones de usar herramientas
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                # 3. Extraemos dinámicamente el nombre y los argumentos nativos
                print(f"The agent wants to execute: {tool_call['name']}")
                print(f"With this data: {tool_call['args']}")
        else:
            print("The agent did not request any tools.")


if __name__ == "__main__":
    asyncio.run(main())