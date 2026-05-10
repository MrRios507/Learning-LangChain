from langchain_ollama import OllamaEmbeddings

model = OllamaEmbeddings(model="embeddinggemma")

embeddings = model.embed_documents([
    "Hi, there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"
])

print(embeddings)