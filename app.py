import streamlit as st
from src.orchestrator import run_agent

st.set_page_config(page_title="Cold-Chain AI Assistant", layout="wide")
st.title("❄️ Autonomous Cold-Chain Dispatcher")

# Session state initialization for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt input
user_prompt = st.chat_input("Enter dispatch instructions or anomaly check...")

if user_prompt:
    # Display user input
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Run agent and display response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing fleet telemetry, corridor weather, and SOPs..."):
            report = run_agent(user_prompt)
            st.markdown(report)
            st.session_state.messages.append({"role": "assistant", "content": report})