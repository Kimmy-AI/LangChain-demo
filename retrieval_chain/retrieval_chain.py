import os
from getpass import getpass
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage


def _getpass(env_var: str):
    if not os.environ.get(env_var):
        os.environ[env_var] = getpass(f"{env_var}=")


_getpass("OPENAI_API_KEY")

loader = WebBaseLoader("https://docs.smith.langchain.com")

docs = loader.load()

# print(docs)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# print(embeddings)

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}"""
)

# print(prompt)

llm = ChatOpenAI()
document_chain = create_stuff_documents_chain(llm, prompt)
# print(document_chain)


document_chain_invoke_res = document_chain.invoke(
    {
        "input": "how can langsmith help with testing?",
        "context": [
            Document(page_content="langsmith can let you visualize test results")
        ],
    }
)
# print(document_chain_invoke_res)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
# print(response["answer"])

# First we need a prompt that we can pass into an LLM to generate this search query
prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
        ),
    ]
)
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

chat_history = [
    HumanMessage(content="Can LangSmith help test my LLM applications?"),
    AIMessage(content="Yes!"),
]
retriever_chain_invoke_res = retriever_chain.invoke(
    {"chat_history": chat_history, "input": "Tell me how"}
)
# print(retriever_chain_invoke_res)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

chat_history = [
    HumanMessage(content="Can LangSmith help test my LLM applications?"),
    AIMessage(content="Yes!"),
]
retriever_chain_invoke_res2 = retrieval_chain.invoke(
    {"chat_history": chat_history, "input": "Tell me how"}
)
print("\n")
print(retriever_chain_invoke_res2)
