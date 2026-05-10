# You can also use an LLM or chat model to produce output in other formats, such a CSV or XML.
# Output parsers are classes that help you structure large language model responses.
# They serve two functions:
# - Providing format instructions: Output parsers can be used to inject some additional instructions in the prompt
# - Validating and parsing output: Take the textual output of the LLM or chat model and render it to a more structured format.
#   This can include removing extraneous information, correcting incomplete output, and validating the parsed values.

from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

items = parser.invoke("apple, banana, cherry")

print(items)