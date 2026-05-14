from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.runnables import chain


connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "rag_docs"

model = OllamaEmbeddings(model="embeddinggemma")

db = PGVector(
    connection=connection,
    embeddings=model,
    collection_name=collection_name,
    use_jsonb=True
)

retriever = db.as_retriever()

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context: {context}
    Question: {question}
""")

llm = ChatOllama(model="granite4.1:3b", temperature=0)

@chain
def qa(input):
    # fetch relevant documents
    docs = retriever.invoke("""Who are the key figures in the ancient greek history of philosophy?""")

    # format prompt
    formatted = prompt.invoke({
        "context": docs, 
        "question": """Who are the key figures in the ancient greek history of philosophy?"""
    })

    # generate answer
    answer = llm.invoke(formatted)

    return {"answer": answer, "docs": docs}

# run
response = qa.invoke("Who are the key figures in the ancient greek history of philosophy?")

print(response)