from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(model="granite4.1:3b")

response = model.invoke("The sky is")

print(response)