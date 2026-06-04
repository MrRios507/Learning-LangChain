# SQL and relational databases are important sources of structured data, but they don't
# interact directly with natural language. Although we can simply use the LLM to translate
# a user's query to SQL queries.

# Here are some useful strategies for effective text to SQL translations:

# - Database description
# To ground SQL queries, an LLM must be provided with an accurate description
# of the database. One common text-to-SQL prompt employs an idea reported in
# this paper and others: provide the LLM with a CREATE TABLE description for
# each table, including column names and types.5 We can also provide a few (for
# instance, three) example rows from the table.

# - Few-shot examples
# Feeding the prompt with few-shot examples of question-query matches can improve the query generation accuracy.

from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_ollama import ChatOllama

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOllama(model="qwen3.5:2b", temperature=0)

# convert question to sql query
write_query = create_sql_query_chain(llm, db)

# Execute SQL Query
execute_query = QuerySQLDataBaseTool(db=db)

# combined
chain = write_query | execute_query

# invoke the chain
result = chain.invoke({"question": "How many employees are there?"})

print(result)