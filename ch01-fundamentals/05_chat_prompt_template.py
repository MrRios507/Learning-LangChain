# The ChatPromptTemplate can be used instead to provide dynamic inputs based on the role of the chat message.
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# both `template` and `model` can be reused many times

template = ChatPromptTemplate.from_messages([
    ('system', '''Answer the question based on the context below. If the question cannot be answered using the information
    provided, answer with "I don\'t know".'''),
    ('human', 'Context: {context}'),
    ('human', 'Question: {question}'),
])

model = ChatOllama(model="granite4.1:3b")

# `prompt` and `completion` are the results of using template and model once
prompt = template.invoke({
    "context": """The most recent advancements in NLP are being driven by Large Language Models (LLMs).
    These models outperform their smaller counterparts and have become invaluable for developers who are creating
    applications with NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library,
    or utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
    "question": "Which model providers offer LLMs?",
})

completion = model.invoke(prompt)

print(completion)