# -*- coding: utf-8 -*- 
# Chatbot using LangGraph, LangChain, and Streamlit
import streamlit as st
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# === Configuration ===
kasioss_url = st.secrets["kasioss"]["kasioss_url"]

# === Setup embeddings & retriever ===
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=st.secrets["openai"]["api_key"] 
)

vectorstore = PGVector(
    connection=kasioss_url+"?sslmode=require",
    embeddings=embeddings,
    collection_name=st.secrets["kasioss"]["collection"],
    use_jsonb=True,
)

# Init LLM
llm = ChatOpenAI(
    model=st.secrets["openai"]["model"],
    temperature=0.1,
    openai_api_key=st.secrets["openai"]["api_key"]
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Pattreeya's assistant who knows best about Pattreeya (Tanisaro).
        You are not able to answer any other questions else. You only know about Pattreeya.
        All related questions are welcome, including her background, work, and education.
        """
    ),
    MessagesPlaceholder(variable_name="messages"),  # <- history from ChatState
    ("user", "{question}")  # last user input
])

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# === Define Chat State ===
class ChatState(TypedDict):
    messages: Annotated[List, add_messages]

# --- Put this near top of file (so callback is top-level) ---
def _on_suggestion_click(sug: str):
    """Callback executed when a suggestion button is clicked."""

    st.session_state.pending_followup = sug
    # trigger a rerun so the pending_followup is processed as a new user question
    st.rerun()

# === Nodes ===
def retrieve_node(state: ChatState):
    """Retrieve documents based on the last user message."""

    query = state["messages"][-1].content
    docs = retriever.invoke(query)

    if not docs:
        return {
            "messages": [AIMessage(content="I couldn’t find anything relevant.")]
        }
    # 
    # Append docs content to system message for the LLM
    docs_text = "\n\n".join([d.page_content for d in docs])
    return {
        "messages": [HumanMessage(content=f"Context:\n{docs_text}")]
    }

def generate_node(state: ChatState):
    """Generate answer from LLM using a ChatPromptTemplate."""

    last_user_message = state["messages"][-1].content

    # Format prompt with history + new question
    formatted = prompt.format_messages(
        messages=state["messages"][:-1],  # all previous messages
        question=last_user_message
    )

    response = llm.invoke(formatted)
    return {"messages": [AIMessage(content=response.content)]}

# === Build Graph ===
workflow = StateGraph(ChatState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()

# === Streamlit UI ===
st.set_page_config(
    page_title="Pattreeya's Chatbot", 
    page_icon=":women:", 
    layout="wide"
)
st.title("Chat with Pattreeya's Assistant")

# === Session State ===
if "graph_state" not in st.session_state:
    st.session_state.graph_state = {"messages": []}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_followup" not in st.session_state:
    st.session_state.pending_followup = None


# === Handle follow-up button click ===
if st.session_state.pending_followup:
    user_question = st.session_state.pending_followup
    st.session_state.pending_followup = None
else:
    user_question = st.chat_input("Ask me anything about Pattreeya...")


 # === Display chat history FIRST ===
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Only assistant messages will have suggestions
        if message.get("suggestions") and message["role"] == "assistant":
            # parse suggestions into a clean list, strip numbering like "1. ..." etc.
            suggestion_list = [
                s.lstrip("0123456789. ").strip()
                for s in message["suggestions"].splitlines()
                if s.strip()
            ]

            # keep at most 2 suggestions (we only support 2 follow-ups)
            if suggestion_list:
                suggestion_list = suggestion_list[:2]

                st.markdown("💡 **Suggested follow-ups:**")

                # create one column per suggestion
                cols = st.columns(len(suggestion_list))

                # render a button in each column using the top-level callback
                for sug_idx, sug in enumerate(suggestion_list):
                    # stable unique key per message+suggestion
                    key = f"suggestion_{idx}_{sug_idx}"

                    # place the button in the correct column and register the callback
                    cols[sug_idx].button(
                        f"{sug_idx+1}. {sug}",
                        key=key,
                        on_click=_on_suggestion_click,
                        args=(sug,),
                    )

# === Handle new user input or follow-up AFTER displaying history ===
if user_question:
    user_msg = HumanMessage(content=user_question)
    st.session_state.graph_state["messages"].append(user_msg)
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_question)

    with st.spinner("Thinking..."):
        try:
            result = graph.invoke(st.session_state.graph_state)
            answer = result["messages"][-1].content

            # Generate follow-up suggestions
            suggestion_prompt = f"""Based on this conversation, suggest 2 brief follow-up questions concerning Pattreeya's background, work and education.
            Refer to the original user question when generating suggestions.
            Focus on her recent experiences, research focus, and skills.
            Use Pattreeya or \"her\" instead of \"you\" in the suggestions but \"you\" refer to the Pattreeya not the user.
            
            Do not repeat the original question.
            
                Question: {user_question}
                Answer: {answer}

            Format as a numbered list."""
            followup_response = llm.invoke(suggestion_prompt)
            suggestions = followup_response.content

            # Display assistant message
            with st.chat_message("assistant"):
                st.write(answer)

                suggestion_list = [
                    s.lstrip("1234567890. ").strip()
                    for s in suggestions.split("\n") if s.strip()
                ]
                if suggestion_list:
                    st.markdown("💡 **Suggested follow-ups:**")
                    cols = st.columns(len(suggestion_list))
                    for idx, sug in enumerate(suggestion_list):
                        if cols[idx].button(f"{idx+1}. {sug}", key=f"new_suggestion_{idx}"):
                            st.session_state.pending_followup = sug
                            st.rerun()

            # Save messages to session state
            st.session_state.graph_state["messages"].append(AIMessage(content=answer))
            st.session_state.messages.append({
                "role": "user",
                "content": user_question
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "suggestions": suggestions
            })
            
            # Rerun to update the display with new messages
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

# === Sidebar ===
with st.sidebar:
    #st.header("Options")
    st.image("https://kasioss.com/pt/images/pt-01.png", width=100, caption="")
    if st.button("Clear Chat History"):
        st.session_state.graph_state = {"messages": []}
        st.session_state.messages = []
        st.session_state.pending_followup = None
        st.rerun()
    
    st.divider()
    st.caption(f"Memory buffer: {len(st.session_state.graph_state['messages'])} messages")
    
    
