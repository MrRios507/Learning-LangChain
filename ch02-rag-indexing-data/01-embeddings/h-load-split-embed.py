from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings

## Load the document
print("Loading document...")

loader = TextLoader("../test.txt")
doc = loader.load()

print("Document Loaded!")


## Split the document

print("Splitting document...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 20,
)
chunks = text_splitter.split_documents(doc)

print("Document splitted!")


## Generate embeddings

print("Embedding documents...")

embeddings_model = OllamaEmbeddings(model="embeddinggemma")
embeddings = embeddings_model.embed_documents(
    [chunk.page_content for chunk in chunks]
)

print(embeddings)