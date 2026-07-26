# The interrupt mode, is the simplest form of control, the user is looking at streaming output
# of the application as it is produced, and manually interrupts it when he sees fit.

# The state is saved as of the last complete step prior to the user hitting the interrupt button.

# The user can choose to:
# - Resume from that point onward.
# - Send new input into the application.
# - Do nothing.
import asyncio
from contextlib import aclosing

from langchain.messages import HumanMessage

from graph import create_graph


async def main():
    graph = create_graph()

    event = asyncio.Event()

    config = {"configurable": {"thread_id": "1"}}

    input = {
        "messages": [
            HumanMessage(
                "How old was the 30th president of the United States when he died?"
            )
        ]
    }

    async with aclosing(graph.astream_events(input, config)) as stream:
        async for chunk in stream:
            if event.is_set():
                break
            else:
                print(chunk)    # do something with the output

    # Simulate interruption after 2 seconds
    await asyncio.sleep(2)
    event.set()


if __name__ == "__main__":
    asyncio.run(main())