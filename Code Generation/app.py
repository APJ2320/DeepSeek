import streamlit as st
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import time
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

import os

# Load environment variables
# Load API key from secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Set it as an environment variable (if required by your code)
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Custom CSS styling
st.markdown(
    """
    <style>
        .main { background-color: #1a1a1a; color: #ffffff; }
        .sidebar .sidebar-content { background-color: #2d2d2d; }
        .stTextInput textarea { color: #ffffff !important; }
        .stSelectbox div[data-baseweb="select"], .stSelectbox svg, .stSelectbox option {
            color: white !important; background-color: #3d3d3d !important;
        }
        div[role="listbox"] div { background-color: #2d2d2d !important; color: white !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Code Companion")

def save_response(response):
    """Saves the AI response to a file."""
    with open('code_generation_output.txt', 'a') as file:
        file.write(response)

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "deepseek-r1:8b"

    new_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1-distill-qwen-32b","deepseek-r1-distill-llama-70b","deepseek-r1:8b", "codellama"],
        index=["deepseek-r1-distill-qwen-32b","deepseek-r1-distill-llama-70b","deepseek-r1:8b", "codellama"].index(st.session_state.selected_model)
    )

    if new_model != st.session_state.selected_model:
        st.session_state.selected_model = new_model
        st.session_state.message_log = [
            {"role": "ai", "content": f"Your AI Pair Programmer with {new_model}. How can I assist you?"}
        ]
        st.rerun()  # Rerun immediately when model is changed

# Select the appropriate model engine dynamically
if st.session_state.selected_model in ["deepseek-r1-distill-qwen-32b","deepseek-r1-distill-llama-70b"]:
    llm_engine = ChatGroq(
        model=st.session_state.selected_model,
        temperature=0,
        verbose=True
    )
else:
    llm_engine = ChatOllama(
        model=st.session_state.selected_model,
        base_url="http://localhost:11434",
        temperature=0.3
    )

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm your code companion. How can I help you code today?"}]

# Chat container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and processing
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt_chain):
    """Processes the prompt chain using the selected LLM engine."""
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    """Builds a prompt chain with system and user messages."""
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    start_time = time.time()
    
    # Generate AI response
    with st.spinner("Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)

    stop_time = time.time()
    
    # Store AI response
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.session_state.message_log.append(
        {"role": "ai", "content": f"Time taken by {st.session_state.selected_model} model: {(stop_time - start_time) / 60:.2f} mins"}
    )

    # Save response to file
    save_response(f"\nModel: {st.session_state.selected_model}")
    save_response(f"\n\nUser Query: {user_query}")
    save_response(f"\n\nResponse:\n\n{ai_response}")
    save_response(f"\n\nTime taken by {st.session_state.selected_model} model: {(stop_time - start_time) / 60:.2f} mins")
    save_response("\n" + "*" * 500)

    # Update chat display
    st.rerun()
