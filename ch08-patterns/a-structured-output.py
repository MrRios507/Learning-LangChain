# Create a schema to use
from pydantic import BaseModel, Field

class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")

# Let's get an LLM to generate output that conforms to this schema
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2:latest", temperature=0)
model = model.with_structured_output(Joke)

result = model.invoke("Tell me a joke about cats")
print(result)