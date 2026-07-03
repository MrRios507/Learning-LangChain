# The chain architecture always executes a static sequence of steps,
# the router architecture is characterized by using an LLM to choose between certain predefined steps.
# The key development in this architecture is to use an LLM to make this decision.

# 1. An LLM call to pick which of the available indexes to use
# - given the user-supplied query
# - the developer-supplied description of the indexes

# 2. A retrieval step that queries the chosen index for the most relevant documents for user query

# 3. Another LLM call to generate an answer
# - given the user-supplied query
# - the list of relevant documents fetched from the index

from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# useful to generate SQL query
model_low_temp = ChatOllama(model="granite4.1:3b", temperature=0.1)

# useful to generate natural language outputs
model_high_temp = ChatOllama(model="granite4.1:3b",temperature=0.7)

class State(TypedDict):
    # to track conversation history
    messages: Annotated[list, add_messages]
    
    # input
    user_query: str
    
    # output
    domain: Literal["records", "insurance"]
    documents: list[Document]
    answer: str

class Input(TypedDict):
    user_query: str

class Output(TypedDict):
    documents: list[Document]
    answer: str

# Fill a vector store with documents
medical_records_store = InMemoryVectorStore.from_documents([], embedding=embeddings)
medical_records_retriever = medical_records_store.as_retriever()

insurance_faqs_store = InMemoryVectorStore.from_documents([], embeddings)
insurance_faqs_retriever = insurance_faqs_store.as_retriever()

router_prompt = SystemMessage(
    """You need to decide which domain to route the user query to. You have two domains to choose from:
        - records: contains medical records of the patient, such as diagnosis, treatment, and prescriptions.
        - insurance: contains frequently asked questions about insurance policies, claims, and coverage.

        Output only the domain name.
    """
)

def router_node(state: State) -> State:
    user_message = HumanMessage(state["user_query"])
    messages = [router_prompt, *state["messages"], user_message]
    res = model_low_temp.invoke(messages)

    return {
        "domain": res.content,
        # update conversation history
        "messages": [user_message, res]
    }

def pick_retriever(state: State) -> Literal["retrieve_medical_records", "retrieve_insurance_faqs"]:
    if state["domain"] == "records":
        return "retrieve_medical_records"
    else:
        return "retrieve_insurance_faqs"
    
def retrieve_medical_records(state: State) -> State:
    documents = medical_records_retriever.invoke(state["user_query"])
    return {
        "documents": documents
    }

def retrieve_insurance_faqs(state: State) -> State:
    documents = insurance_faqs_retriever.invoke(state["user_query"])
    return {
        "documents": documents
    }

medical_records_prompt = SystemMessage(
    """You are a helpful medical chatbot who answers questions based on the patient's medical records,
        such a diagnosis, treatment, and prescriptions."""
)

insurance_faqs_prompt = SystemMessage(
    """You are a helpful medical insurance chatbot who answers frequently asked
        questions about insurance policies, claims, and coverage."""
)

def generate_answer(state: State) -> State:
    if state["domain"] == "records":
        prompt = medical_records_prompt
    else:
        prompt = insurance_faqs_prompt
    messages = [
        prompt,
        *state["messages"],
        HumanMessage(f"Documents: {state["documents"]}"),
    ]
    res = model_high_temp.invoke(messages)

    return {
        "answer": res.content,
        # update conversation history
        "messages": res,
    }

builder = StateGraph(State, input_schema=Input, output_schema=Output)

builder.add_node("router", router_node)
builder.add_node("retrieve_medical_records", retrieve_medical_records)
builder.add_node("retrieve_insurance_faqs", retrieve_insurance_faqs)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", pick_retriever)
builder.add_edge("retrieve_medical_records", "generate_answer")
builder.add_edge("retrieve_insurance_faqs", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()

# Draw a visual representation of the graph
graph_png = graph.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(graph_png)

input = {
    "user_query": "Am I covered for COVID-19 treatment?"
}
for c in graph.stream(input):
    print(c)