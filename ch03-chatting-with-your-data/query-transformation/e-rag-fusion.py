# The RAG-Fusion strategy shares similarities with the multi-query retrieval strategy, except we will
# apply a final reranking step to all the retrieved documents.
# Reciprocal Rank Fusion (RRF) algorithm

# RAG-Fusion's strength lies in its ability to capture the user's intented expression, 
# navigate complex queries, and broaden the scope of retrieved documents, enabling serendipitous discovery.


from langchain_core.runnables import chain
from langchain_core.prompts import ChatPromptTemplate
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

prompt_rag_fusion = ChatPromptTemplate.from_template("""
    You are helpful assistant that generates multiple search queries based on single input query.\n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):
""")

def parse_queries_output(message):
    return message.content.split('\n')

llm = ChatOllama(model="granite4.1:3b", temperature=0)

query_gen = prompt_rag_fusion | llm | parse_queries_output

def reciprocal_rank_fusion(results: list[list], k=60):
    """reciprocal rank fusion on multiple list of ranked documents 
    and an optional parameter k used in the RRF formula"""
    
    # Initialize a dictionary to hold fused scores for each document
    # Documents will be keyed by their contents to ensure uniqueness
    fused_scores = {}
    documents = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Use the document contents as the key for uniqueness
            doc_str = doc.page_content
            # If the document hasn't been seen yet:
            # - initialize score to 0
            # - save it for later
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc
                
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_doc_strs = sorted(fused_scores, key=lambda d: fused_scores[d], reverse=True)

    # Retrieve the corresponding doc for each doc_str
    return [
        documents[doc_str]
        for doc_str in reranked_doc_strs
    ]

retrieval_chain = query_gen | retriever.batch | reciprocal_rank_fusion

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