# When generating JSON, the first task is to define the schema you want the LLM to respect when producing the output.
# Then, you should include that schema in the prompt, along with the text you want to use as the source.

from langchain_ollama import ChatOllama
from pydantic import BaseModel

class AnswerWithJustification(BaseModel):
    '''An answer to the user's question along with justification for the answer.'''
    answer: str
    '''The answer to the user's question'''
    justification: str
    '''Justification for the answer'''

llm = ChatOllama(model="granite4.1:3b", temperature=0)

structured_llm = llm.with_structured_output(AnswerWithJustification)

response = structured_llm.invoke("""What weighs more, a pound of bricks or a pound of feathers""")

print(response)