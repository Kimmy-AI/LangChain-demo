from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model_name="gpt-4.0-turbo", base_url="https://lonlie.plus7.plus/v1")

# print(llm.invoke("how can langsmith help with testing?"))

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are my helpful assistant."), ("user", "{input}")]
)

# print(prompt)

chain = prompt | llm

# print(chain)

# print(chain.invoke({"input": "how can langsmith help with testing?"}))

output_parser = StrOutputParser()

# print(output_parser)

chain = prompt | llm | output_parser

# print(chain)

print(chain.invoke({"input": "how are you?"}))
