# Logical Routing
# In logical routing, we give the LLM knowledge of the various data sources at our disposal,
# and then let the LLM reason which data source to apply based on the user's query.

# A function call involves defining a schema that the model can use to generate arguments of a
# function based on the query. This enables us to generate structured outputs that can be used to run other functions.

from pydantic import BaseModel, Field
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda

# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["python_docs", "js_docs"] = Field(
        ...,
        description="""Given a user question, choose which datasource would be most relevant for answering their question""",
    )

# LLM with function call
llm = ChatOllama(model="granite4.1:3b", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)

# Prompt
system = """
    You are an expert at routing a user question to the appropriate data source.
    
    Based on the programming language the question is referring to, route it to the relevant data source.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Define router
router = prompt | structured_llm

question = """Why doesn't the following code work: 
from langchain_core.prompts 
import ChatPromptTemplate 
prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"]) 
prompt.invoke("french")
"""
result = router.invoke({"question": question})
print("\nRouting to: ", result)

# Once we've extracted the relevant data source, we can pass the value into another funciton to execute additional logic as required
def choose_route(result):
    if "python_docs" in result.datasource.lower():
        ### Logic here
        return "chain for python_docs"
    else:
        ### Logic here
        return "chain for js_docs"
    
full_chain = router | RunnableLambda(choose_route)

result = full_chain.invoke({"question": question})
print("\nChoose route: ", result)