from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# Initialize Ollama from local
llm = ChatOllama(
    model="llama3.2:3b",       
    temperature=0,
    # base_url="http://localhost:11434",   # thường không cần nếu để mặc định
)

# Prompt 
prompt = ChatPromptTemplate.from_template(
    "Say hi to user and introduce you are {name}."
)

# Make chain
chain = prompt | llm

# Test
response = chain.invoke({"name": "Hoang's RAG bot"})
print(response.content)