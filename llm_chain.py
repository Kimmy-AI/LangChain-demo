from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# fill your openai_api_key
llm = ChatOpenAI(openai_api_key="")

# print(llm.invoke("how can langsmith help with testing?"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are my lovely girl friend."),
    ("user", "{input}")
])

# print(prompt)

chain = prompt | llm 

# print(chain)

# print(chain.invoke({"input": "how can langsmith help with testing?"}))

output_parser = StrOutputParser()

# print(output_parser)

chain = prompt | llm | output_parser

# print(chain)

print(chain.invoke({"input": "how are you?"}))
