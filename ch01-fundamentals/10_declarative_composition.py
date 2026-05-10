# Declarative Composition
# LangChain Expression Language (LCEL)
# LCEL is a declarative language for composing LangChain components. LangChain compiles LCEL compositions to an optimized
# execution plan, with automatic parallelization, streaming, tracing, and async support.

from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# the building blocks

template = ChatPromptTemplate.from_messages([
    ('system', '''Answer the question based on the context below. If the question cannot be answered using the information
    provided, answer with "I don\'t know".'''),
    ('human', 'Context: {context}'),
    ('human', 'Question: {question}'),
])

model = ChatOllama(model="granite4.1:3b")

# combine them with the | operator

chatbot = template | model

# use it
response = chatbot.invoke({
    "context": """The most recent advancements in NLP are being driven by Large Language Models (LLMs).
    These models outperform their smaller counterparts and have become invaluable for developers who are creating
    applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library,
    or utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
    "question": "Which model providers offer LLMs?"
})


print(response)

# You don't need to do anything else to use streaming
for part in chatbot.stream({
    "context": """The most recent advancements in NLP are being driven by Large Language Models (LLMs).
    These models outperform their smaller counterparts and have become invaluable for developers who are creating
    applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library,
    or utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
    "question": "Which model providers offer LLMs?"
}):
    print(part)
