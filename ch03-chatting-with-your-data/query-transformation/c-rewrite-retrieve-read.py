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

# create retriever to retrieve 2 relevant documents
retriever = db.as_retriever(search_kwargs={"k": 2})

# Query starts with irrelevant information before asking the relevant question
query = 'Today I woke up and brushed my teeth, then I sat down to read the news. But then I forgot the food on the cooker. Who are some key figures in the ancient greek history of philosophy?'

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context: {context}
    Question: {question}
""")

llm = ChatOllama(model="granite4.1:3b", temperature=0)

rewrite_prompt = ChatPromptTemplate.from_template(
    """Provide a better search query for web search engine to answer the given question, end the queries with ’**’. Question: {x} Answer:""")


def parse_rewriter_output(message):
    return message.content.strip('"').strip("**")

rewriter = rewrite_prompt | llm | parse_rewriter_output

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



@chain
def qa_rrr(input):
    # rewrite the query
    new_query = rewriter.invoke(input)
    print("Rewritten query: ", new_query)

    # fetch relevant documents
    docs = retriever.invoke(new_query)

    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})

    answer = llm.invoke(formatted)

    return answer



# run
# response = qa.invoke("Who are the key figures in the ancient greek history of philosophy?")
result = qa_rrr.invoke(query)

print(result.content)