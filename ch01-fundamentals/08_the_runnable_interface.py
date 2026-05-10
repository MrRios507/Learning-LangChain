# invoke: transforms a single input into an output.
# batch: effeciently transforms multiple inputs into multiple outputs
# stream: streams output from a single input as it' produced

from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(model="granite4.1:3b")

completion = model.invoke('Hi There!')
print(completion)

completions = model.batch(['Hi There!', 'Bye!'])
print(completions)

for token in model.stream('Bye!'):
    print(token)