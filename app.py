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
    temperature=0.2,  # Increased for more creative synthesis while maintaining accuracy
    openai_api_key=st.secrets["openai"]["api_key"]
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Pattreeya's assistant. You MUST follow these rules:

FACTUAL ACCURACY (Non-Negotiable):
1. For contact information (email, LinkedIn, GitHub, phone, address):
   - Copy ALL fields EXACTLY as they appear in the RETRIEVED CONTEXT
   - CRITICAL: Copy emails CHARACTER-BY-CHARACTER (e.g., "ptanisaro" NOT "patnisaro")
   - Include BOTH "email" and "email-alt" fields if both exist
   - Read emails TWICE before writing them
2. For dates, company names, job titles, education degrees, publication titles:
   - Copy EXACTLY as shown in the RETRIEVED CONTEXT
   - Do NOT approximate or paraphrase

CREATIVE SYNTHESIS (Encouraged):
When discussing Pattreeya's work and research, you may:
- Draw connections between her different experiences and skills
- Explain how her background in one area (e.g., navigation systems) relates to her current work (e.g., LLMs and RAG)
- Highlight patterns across her career (e.g., focus on time-series analysis, from human motion to agricultural data)
- Discuss how her academic research (PhD on time-dependent data) informs her industry work
- Synthesize insights from multiple projects and positions
- Use analogies to explain her technical expertise to different audiences

SCOPE OF QUESTIONS:
- You are ONLY designed to answer questions about Pattreeya Tanisaro
- If asked about unrelated topics, respond: "I appreciate your question, but I'm specifically designed to answer questions about Pattreeya Tanisaro only. Please feel free to ask me anything about her background, work experience, education, skills, publications, or research!"

RESPONSE STYLE:
- Be insightful and make meaningful connections between her experiences
- Show the evolution and continuity in her career path
- Highlight transferable skills and deep expertise
- When appropriate, connect her academic research to practical applications

LANGUAGE MATCHING (Critical):
- ALWAYS respond in the SAME LANGUAGE the user uses to ask the question
- If the user asks in German, respond in German
- If the user asks in English, respond in English
- If the user asks in Thai, respond in Thai
- Detect the language from the user's question and match it exactly
- This applies to ALL responses, including follow-up suggestions

Remember: Be factually precise with data points, but intellectually creative in showing relationships and insights.
        """
    ),
    MessagesPlaceholder(variable_name="messages"),  # <- history from ChatState
    ("user", "{question}")  # last user input
])

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

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

    query = state["messages"][-1].content.lower()
    docs = retriever.invoke(query)

    if not docs:
        return {
            "messages": [AIMessage(content="I couldn't find anything relevant.")]
        }

    # Check if query is about contact information
    contact_keywords = ["email", "contact", "phone", "linkedin", "github", "reach", "chatbot", "how to reach", "get in touch"]
    is_contact_query = any(keyword in query for keyword in contact_keywords)

    # If it's a contact query, ALWAYS ensure CV metadata is at the top
    if is_contact_query:
        # Fetch CV metadata directly by ID to ensure we get the correct contact info
        try:
            # Try to get the metadata document directly
            metadata_docs = vectorstore.similarity_search("contact information", k=2, filter={"type": "cv_metadata"})

            # Prioritize the cv_metadata document
            cv_metadata = [d for d in metadata_docs if d.metadata.get("type") == "cv_metadata"]
            other_docs = [d for d in docs if d.metadata.get("type") != "cv_metadata"]

            if cv_metadata:
                # Put CV metadata first, then other relevant docs
                docs = cv_metadata[:1] + other_docs[:4]
            else:
                # If we couldn't find cv_metadata, still try to get it from the original docs
                has_metadata = any(d.metadata.get("type") == "cv_metadata" for d in docs)
                if not has_metadata:
                    docs = metadata_docs[:1] + docs[:4]
        except Exception:
            # If anything fails, continue with original docs
            pass

    # Format context with clear labeling
    docs_text = "=== RETRIEVED CONTEXT FROM DATABASE ===\n\n"
    for i, d in enumerate(docs, 1):
        section = d.metadata.get("section", "general")
        content = d.page_content

        # If this is cv_metadata with contact info, highlight emails clearly
        if d.metadata.get("type") == "cv_metadata" and "contact" in d.metadata:
            contact = d.metadata["contact"]
            docs_text += f"[Source {i} - {section}]\n"
            docs_text += f"⚠️ CONTACT INFORMATION - COPY EXACTLY CHARACTER-BY-CHARACTER:\n"
            if "email" in contact:
                docs_text += f"  PRIMARY EMAIL: {contact['email']}\n"
            if "email-alt" in contact:
                docs_text += f"  ALTERNATIVE EMAIL: {contact['email-alt']}\n"
            if "linkedin" in contact:
                docs_text += f"  LINKEDIN: {contact['linkedin']}\n"
            if "github" in contact:
                docs_text += f"  GITHUB: {contact['github']}\n"
            if "chatbot" in contact:
                docs_text += f"  CHATBOT: {contact['chatbot']}\n"
            docs_text += f"\nFull metadata:\n{content}\n\n"
        else:
            docs_text += f"[Source {i} - {section}]\n{content}\n\n"

    docs_text += "=== END OF RETRIEVED CONTEXT ===\n"

    return {
        "messages": [HumanMessage(content=docs_text)]
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
            suggestion_prompt = f"""Based on this conversation about Pattreeya Tanisaro, suggest 2 insightful follow-up questions that explore:

1. Her professional experience and technical expertise
2. Her research work and academic contributions
3. Connections between her different roles and projects
4. Evolution of her skills from academia to industry
5. How her past work informs her current capabilities

Guidelines:
- Make questions thought-provoking and specific (not generic)
- Focus on experience, research, publications, or skill development
- Build on the context of the current conversation
- Use "her" or "Pattreeya" (not "you")
- Keep questions concise (under 15 words each)
- Do NOT repeat the original question
- CRITICAL: Generate follow-up questions in the SAME LANGUAGE as the user's question
  (If question is in German, suggestions in German; if English, suggestions in English; if Thai, suggestions in Thai)

Examples of good follow-ups:
- "How did her navigation systems work influence her current RAG implementations?"
- "What research insights from her PhD apply to agricultural data analysis?"
- "How has her time-series expertise evolved from academia to industry?"

Current conversation:
Question: {user_question}
Answer: {answer}

Format as a numbered list (1. and 2.)"""
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
    st.image(st.secrets["kasioss"]["profile_pic"], width=100, caption="")
    if st.button("Clear Chat History"):
        st.session_state.graph_state = {"messages": []}
        st.session_state.messages = []
        st.session_state.pending_followup = None
        st.rerun()
    
    st.divider()
    st.caption(f"Memory buffer: {len(st.session_state.graph_state['messages'])} messages")
    
    
