# Multi-Query Retrieval
# A user's single query can be insufficient to capture the full scope of information required to answer the query comprehensively.
# The multi-query retrieval resolves this problem by instructing an LLM to generate multiple queries based on a user's initial query.

from langchain_core.runnables import chain
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

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


perspectives_prompt = ChatPromptTemplate.from_template(
    """
    You are an AI language model assistant.
    Your task is to generate three different versions of the given user question to retrieve relevant documents from a vector database.
    By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines. Original question: {question}
""")

def parse_queries_output(message):
    print("Perspective prompt: ", message)
    return message.content.split('\n')

query_gen = perspectives_prompt | llm | parse_queries_output


def get_unique_union(document_lists):
    print("Document List: ", document_lists)
    # Flatten list of lists, and dedupe them
    deduped_docs = { doc.page_content : doc for sublist in document_lists for doc in sublist }

    # return a flat list of unique_docs
    return list(deduped_docs.values())

retrieval_chain = query_gen | retriever.batch | get_unique_union


prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)

query = "Who are the key figures in the ancient greek history of philosophy?"

@chain
def multi_query_qa(input):
    # fetch relevant documents
    docs = retrieval_chain.invoke(input)  # format prompt
    formatted = prompt.invoke(
        {"context": docs, "question": input})  # generate answer
    answer = llm.invoke(formatted)
    return answer


# run
print("Running multi query qa\n")
result = multi_query_qa.invoke(query)
print(result.content)