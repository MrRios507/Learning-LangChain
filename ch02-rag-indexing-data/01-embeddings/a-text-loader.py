from langchain_community.document_loaders import TextLoader

loader = TextLoader("../test.txt")

result = loader.load()

print(result)