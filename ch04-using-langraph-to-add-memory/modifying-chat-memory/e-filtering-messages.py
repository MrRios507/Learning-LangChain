# LangChain's filter_messages helper makes it easier to filter chat history by type, ID, or name.

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    filter_messages,
)

messages = [
    SystemMessage("you are a good assistant", id="1"),
    HumanMessage("example input", id="2", name="example_user"),
    AIMessage("example output", id="3", name="example_assistant"),
    HumanMessage("real input", id="4", name="bob"),
    AIMessage("real output", id="5", name="alice"),
]

# Filter messages by human type
filter_1 = filter_messages(
    messages=messages,
    include_types="human"
)
print(filter_1)


print(110*"=")
# Exclude messages by names
filter_2 = filter_messages(
    messages=messages,
    exclude_names=["example_user", "example_assistant"]
)
print(filter_2)

print(110*"=")
# Include messages by type
filter_3 = filter_messages(
    messages=messages,
    include_types=[HumanMessage, AIMessage],
    exclude_ids=["3"]
)
print(filter_3)
