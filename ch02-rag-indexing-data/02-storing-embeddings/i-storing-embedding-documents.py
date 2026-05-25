from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
import uuid

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

# Load the document, split it into chunks
raw_documents = TextLoader('../../test.txt').load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = text_splitter.split_documents(raw_documents)

# Embed each chunk and insert it into the vector store
embeddings_model = OllamaEmbeddings(model="embeddinggemma")

db = PGVector.from_documents(documents, embeddings_model, connection=connection)

results = db.similarity_search("query", k=4)

print(results)

print("Adding documents to the vector store")
ids = [str(uuid.uuid4()), str(uuid.uuid4())]
db.add_documents(
    [
        Document(
            page_content="there are cats in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
        Document(
            page_content="ducks are also found in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
    ],
    ids=ids,
)

print("Documents added successfully.\n Fetched documents count:",
      len(db.get_by_ids(ids)))

print("Deleting document with id", ids[1])
db.delete({"ids": ids})

print("Document deleted successfully.\n Fetched documents count:",
      len(db.get_by_ids(ids)))