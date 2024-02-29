# LangChain-demo
Some examples built by LangChain and LangGraph

## Introduction
1. `agent.py`  
Use tools from LangChain and OpenAI to create an example of an agent based on language understanding. The main steps include loading documents, creating retriever, setting up tools, etc. Agent can answer user questions about LangSmith and other topics

2. `llmchain.py`  
LCEL usage examples make it easy to build complex chains from basic components

3. `retrieval_chain.py`  
This project demonstrates the implementation of a chatbot and document retrieval system using LangChain libraries and OpenAI API. The system is designed to answer questions based on provided context and historical conversation.

4. `server.py & client.py`  
This serve.py project sets up a FastAPI server using LangChain's Runnable interfaces to interact with LangChain's ChatOpenAI agent and tools. You can run client.py to send request to the server and receive response

5. `web_voyager.py`  
This project demonstrates an interactive agent that uses Playwright to interact with a web page and generate responses based on user input and web page content using OpenAI's GPT-4 model. Agents can perform operations such as clicking elements, typing text, scrolling, waiting, navigating back, and visiting specific web pages, and loop through sampling, parsing, thinking, acting, and storing links

## Usage
1. Set environment variables:
```
export TAVILY_API_KEY=
export OPENAI_API_KEY=
```

2. Install and run the code:
```
pip3 install -r requirements.txt 
python3 agent/agent.py
```