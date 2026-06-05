from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_google_genai import ChatGoogleGenerativeAI

# The parameter strategy controls whether to start from the beginning or the end
# of the list. Usually, you’ll want to prioritize the most recent messages and cut older
# messages if they don’t fit.

# The token_counter is an LLM or chat model, which will be used to count tokens
# using the tokenizer appropriate to that model.

# include_system=True to ensure that the trimmer keeps the system message.

# The parameter allow_partial determines whether to cut the last message’s content to fit within the limit. 

# The parameter start_on="human" ensures that we never remove an AIMessage without also removing a corresponding HumanMessage

trimmer = trim_messages(
    max_tokens=30,
    strategy="last",
    token_counter=ChatGoogleGenerativeAI(model="gemini-3-flash-preview"),
    include_system=True,
    allow_partial=False,
    start_on="human"
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="what's 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

result = trimmer.invoke(messages)
print(result)