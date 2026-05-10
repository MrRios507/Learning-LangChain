from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../test.pdf")

result = loader.load()

print(result)