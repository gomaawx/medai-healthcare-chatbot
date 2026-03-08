import streamlit as st
from medai_bot import SessionState, bot_reply, menu

st.set_page_config(
    page_title="MedAI Assistant",
    page_icon="🏥"
)

st.title("🏥 MedAI Virtual Health Assistant")

if "state" not in st.session_state:
    st.session_state.state = SessionState()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Welcome to MedAI.\n\n" + menu()}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Type your message")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    reply = bot_reply(prompt, st.session_state.state)

    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply)