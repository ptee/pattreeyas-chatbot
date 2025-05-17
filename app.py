# -*- coding: utf-8 -*- 
# Simple chatbot using LangChain and Streamlit
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_postgres.vectorstores import PGVector
#from langchain_postgres import PGVectorStore, PGEngine, PGVector


# NEON.tech
neon_url = st.secrets["neon"]["documents_pt_url"]


# Init Embeddings model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=st.secrets["openai"]["api_key"] 
)

# Connect embeddings to neon database
vector_store = PGVector(
    connection=neon_url+"?sslmode=require",
    embeddings=embeddings,
    collection_name="documents_pt",
    use_jsonb=True,
)

# Init LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0 
)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a chatbot and a helpful assistant who knows best about Pattreeya Tanisaro."),
        ("human", "{input}"),
        ("ai", "{chat_history}"), # Placeholder for chat history
        ("placeholder", "{agent_scratchpad}"), # Agent's thought process
    ]
)


# Create the retriever tool
@tool("retrieve_db")
def retrieve_db(query: str):
    """ Retrieve information related to a question."""

    retrieved_docs = vector_store.similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Combine all tools
tools = [ retrieve_db ]

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Init streamlit app page config
st.set_page_config(page_title="Pattreeya's Chatbot", page_icon=":women:") 

# Set the title and image
col1, col2 = st.columns([10, 2])
with col1:
    st.title("Ask me anything about Pattreeya")
    
with col2:
    st.image("https://kasioss.com/pt/images/pt-01.png", width=100, caption="")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Create an input where we can type messages
user_question = st.chat_input("Ask me anything about Pattreeya.")

# User submit a question?
if user_question:

    # Add the message from the user to the screen+chat with streamlit
    with st.chat_message("user"):
        st.markdown( user_question )
        st.session_state.messages.append( HumanMessage(user_question) )

    # Invoke the agent
    result = agent_executor.invoke(
        { "input": user_question, 
          "chat_history": st.session_state.messages
        }
    )

    ai_message = result["output"]

    # Add the response from LLm to the screen+chat
    with st.chat_message("assistant"):
        st.markdown(ai_message)
        st.session_state.messages.append( AIMessage(ai_message) )

############################ END OF FILE #####################################