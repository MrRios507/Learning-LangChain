# Imperative Composition
# It is just a fancy name for writing the code you're used to writing, composing these components into functions and classes.
# An example combining prompts, models, and output parsers.

from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

# the building blocks
template = ChatPromptTemplate.from_messages([
    ('system', '''Answer the question based on the context below. If the question cannot be answered using the information
    provided, answer with "I don\'t know".'''),
    ('human', 'Context: {context}'),
    ('human', 'Question: {question}'),
])

model = ChatOllama(model="granite4.1:3b")

# combine them in a function
# @chain decorator adds the same Runnable interface for any function you write

@chain
def chatbot(values):
    prompt = template.invoke(values)
    return model.invoke(prompt)

response = chatbot.invoke({
    "context": """The most recent advancements in NLP are being driven by Large Language Models (LLMs).
    These models outperform their smaller counterparts and have become invaluable for developers who are creating
    applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library,
    or utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
    "question": "Which model providers offer LLMs?"
})


print(response)

# Enable streaming or async support
@chain
def chatbot_streaming(values):
    prompt = template.invoke(values)
    for token in model.stream(prompt):
        yield token

for part in chatbot_streaming.stream({
    "context": """The most recent advancements in NLP are being driven by Large Language Models (LLMs).
    These models outperform their smaller counterparts and have become invaluable for developers who are creating
    applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library,
    or utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
    "question": "Which model providers offer LLMs?"
}):
    print(part)


# For asynchronous execution
@chain
async def chatbot_async(values):
    prompt = await template.ainvoke(values)
    return await model.ainvoke(prompt)

chatbot_async.invoke({
    "context": """The most recent advancements in NLP are being driven by Large Language Models (LLMs).
    These models outperform their smaller counterparts and have become invaluable for developers who are creating
    applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library,
    or utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
    "question": "Which model providers offer LLMs?"
})