# Hypothetical Document Embeddings (HyDE) is a strategy that involves creating a hypothetical document based on the user's query,
# embedding the document, and retrieving relevant documents based on vector similarity.

from langchain_core.runnables import chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "rag_docs"

model = OllamaEmbeddings(model="embeddinggemma")
llm = ChatOllama(model="granite4.1:3b", temperature=0)

db = PGVector(
    connection=connection,
    embeddings=model,
    collection_name=collection_name
)

# create retriever to retrieve 3 relevant documents
retriever = db.as_retriever(search_kwargs={"k": 3})

# First, define a prompt to generate a hypothetical document
prompt_hyde = ChatPromptTemplate.from_template("""
    Please write a passage to answer the question.
    Question: {question}
    Passage:""")

generate_doc = (
    prompt_hyde |
    ChatOllama(model="granite4.1:3b", temperature=0) |
    StrOutputParser()
)

# Next, we take the hypothetical document and use it as input to the retriever.
# Which will generate its embedding and search for similar documents in the vector store
retrieval_chain = generate_doc | retriever


# Finally, we take the retrieved documents, pass them as context to the final prompt, and
# instruct the model to generate an output
prompt = ChatPromptTemplate.from_template("""
    Answert the following question based on this context:
    {context}
    Question: {question}
""")

@chain
def qa(input):
    # fetch relevant documents from the hyde retrieval chain defined earlier
    docs = retrieval_chain.invoke(input)
    
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})

    # generate answer
    answer = llm.invoke(formatted)

    return answer

result = qa.invoke("""Who are some key figures in the ancient greek history of philosophy?""")
print(result)