from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama

model = ChatOllama(model="granite4.1:3b")

system_msg = SystemMessage(
    '''You are a helpful assistant that responds to questions with three exclamation marks.'''
)
human_msg = HumanMessage('What is the capital of France?')

response = model.invoke([system_msg, human_msg])

print(response)