from langchain_classic.indexes import SQLRecordManager, index
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "rag_docs"
name_space = "rag_docs_namespace"

# Load the document, split it into chunks
raw_documents = TextLoader('../test.txt').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

documents = text_splitter.split_documents(raw_documents)

# Embed each chunk and insert it into the vector store
model = OllamaEmbeddings(model="embeddinggemma")

vectorstore = PGVector(
    embeddings=model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True
)

record_manager = SQLRecordManager(
    namespace=name_space,
    db_url="postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
)

record_manager.create_schema()

index(
    documents,
    record_manager=record_manager,
    vector_store=vectorstore,
    cleanup="incremental", # prevent duplicate documents
    source_id_key="source" # use the source field as the source_id
)

# create retriever
retriever = vectorstore.as_retriever()

# fetch relevant documents
docs = retriever.invoke("""Who are the key figures in the ancient greek history of philosophy?""")
print("First retriever")
print(docs)

# create a retriever with k=2
retriever_k2 = vectorstore.as_retriever(search_kwargs={"k": 2})

# fetch the 2 most relevant documents
docs_k2 = retriever_k2.invoke("""Who are the key figures in the ancient greek history of philosophy?""")
print("Second retriever")
print(docs_k2)

